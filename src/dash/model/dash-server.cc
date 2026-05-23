/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 TEI of Western Macedonia, Greece
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Dimitrios J. Vergados <djvergad@gmail.com>
 */

#include "dash-server.h"

#include "http-header.h"
#include "mpeg-header.h"

#include "ns3/address-utils.h"
#include "ns3/address.h"
#include "ns3/inet-socket-address.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/log.h"
#include <sstream>
#include <fstream>
#include "ns3/node.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/socket-factory.h"
#include "ns3/socket.h"
#include "ns3/string.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/udp-socket.h"
#include <ns3/double.h>
#include <ns3/random-variable-stream.h>
#include <ns3/tcp-socket.h>

namespace
{
bool
IsLadderBitrate(uint32_t bitrateBps)
{
    static const uint32_t ladder[] = {250000, 750000, 1500000, 3000000, 6000000, 12000000};
    for (uint32_t rate : ladder)
    {
        if (rate == bitrateBps)
        {
            return true;
        }
    }
    return false;
}
} // namespace

namespace ns3
{
NS_LOG_COMPONENT_DEFINE("DashServer");
NS_OBJECT_ENSURE_REGISTERED(DashServer);

TypeId
DashServer::GetTypeId(void)
{
    static TypeId tid = TypeId("ns3::DashServer")
                            .SetParent<Application>()
                            .AddConstructor<DashServer>()
                            .AddAttribute("Local",
                                          "The Address on which to Bind the rx socket.",
                                          AddressValue(),
                                          MakeAddressAccessor(&DashServer::m_local),
                                          MakeAddressChecker())
                            .AddAttribute("Protocol",
                                          "The type id of the protocol to use for the rx socket.",
                                          TypeIdValue(TcpSocketFactory::GetTypeId()),
                                          MakeTypeIdAccessor(&DashServer::m_tid),
                                          MakeTypeIdChecker())
                            .AddAttribute("MetadataCsvPath",
                                          "Path to metadata CSV with size_bytes_* columns.",
                                          StringValue(""),
                                          MakeStringAccessor(&DashServer::m_metadataCsvPath),
                                          MakeStringChecker())
                            .AddAttribute("MetadataTracePath",
                                          "Optional CSV trace of per-segment payload source/size.",
                                          StringValue(""),
                                          MakeStringAccessor(&DashServer::m_metadataTracePath),
                                          MakeStringChecker())
                            .AddTraceSource("Rx",
                                            "A packet has been received",
                                            MakeTraceSourceAccessor(&DashServer::m_rxTrace),
                                            "ns3::Packet::TracedCallback");
    return tid;
}

DashServer::DashServer()
    : m_socket(0),
      m_totalRx(0),
      m_metadataReady(false),
      m_metadataTraceHeaderWritten(false)
{
    NS_LOG_FUNCTION(this);
}

DashServer::~DashServer()
{
    NS_LOG_FUNCTION(this);
}

Ptr<Socket>
DashServer::GetListeningSocket(void) const
{
    NS_LOG_FUNCTION(this);
    return m_socket;
}

std::list<Ptr<Socket>>
DashServer::GetAcceptedSockets(void) const
{
    NS_LOG_FUNCTION(this);
    return m_socketList;
}

void
DashServer::DoDispose(void)
{
    NS_LOG_FUNCTION(this);
    m_socket = 0;
    m_socketList.clear();
    m_pendingPackets.clear();
    m_queues.clear();

    // chain up
    Application::DoDispose();
}

// Application Methods
void
DashServer::StartApplication() // Called at time specified by Start
{
    NS_LOG_FUNCTION(this);
    if (!m_metadataReady && !m_metadataCsvPath.empty())
    {
        std::string error;
        if (m_metadataIndex.Load(m_metadataCsvPath, &error))
        {
            m_metadataReady = true;
            NS_LOG_UNCOND("DashServer loaded metadata CSV: " << m_metadataCsvPath
                          << " segments=" << m_metadataIndex.GetSegmentCount());
        }
        else
        {
            NS_LOG_WARN("DashServer metadata load failed (" << error
                         << "). Falling back to legacy synthetic segment sizes.");
        }
    }
    else if (m_metadataCsvPath.empty())
    {
        NS_LOG_UNCOND("DashServer metadata CSV path is empty; using synthetic segment sizes.");
    }

    if (!m_metadataTracePath.empty())
    {
        NS_LOG_UNCOND("DashServer metadata trace enabled: " << m_metadataTracePath);
    }

    // Create the socket if not already
    if (!m_socket)
    {
        m_socket = Socket::CreateSocket(GetNode(), m_tid);
        m_socket->Bind(m_local);
        m_socket->Listen();
        // m_socket->ShutdownSend ();
        if (addressUtils::IsMulticast(m_local))
        {
            Ptr<UdpSocket> udpSocket = DynamicCast<UdpSocket>(m_socket);
            if (udpSocket)
            {
                // equivalent to setsockopt (MCAST_JOIN_GROUP)
                udpSocket->MulticastJoinGroup(0, m_local);
            }
            else
            {
                NS_FATAL_ERROR("Error: joining multicast on a non-UDP socket");
            }
        }
    }

    m_socket->SetRecvCallback(MakeCallback(&DashServer::HandleRead, this));

    m_socket->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                MakeCallback(&DashServer::HandleAccept, this));
    m_socket->SetCloseCallbacks(MakeCallback(&DashServer::HandlePeerClose, this),
                                MakeCallback(&DashServer::HandlePeerError, this));
}

void
DashServer::StopApplication() // Called at time specified by Stop
{
    NS_LOG_FUNCTION(this);
    while (!m_socketList.empty()) // these are accepted sockets, close them
    {
        Ptr<Socket> acceptedSocket = m_socketList.front();
        m_socketList.pop_front();
        acceptedSocket->Close();
    }
    if (m_socket)
    {
        m_socket->Close();
        m_socket->SetRecvCallback(MakeNullCallback<void, Ptr<Socket>>());
    }
    m_pendingPackets.clear();
    m_queues.clear();
}

void
DashServer::HandleRead(Ptr<Socket> socket)
{
    NS_LOG_FUNCTION(this << socket);
    
    Ptr<Packet> packet;
    Address from;

    while ((packet = socket->RecvFrom(from)))
    {
        m_totalRx += packet->GetSize();

        Ptr<Packet>& pendingPacket = m_pendingPackets[socket];

        if (!pendingPacket)
        {
            pendingPacket = packet;
        }
        else
        {
            pendingPacket->AddAtEnd(packet);
        }

        HTTPHeader header;
        uint32_t httpHeaderSize = header.GetSerializedSize();

        while (pendingPacket->GetSize() >= httpHeaderSize)
        {
            // Validate packet has enough data before removing header
            if (pendingPacket->GetSize() < httpHeaderSize)
            {
                NS_LOG_ERROR("Packet size (" << pendingPacket->GetSize()
                             << ") is smaller than HTTP header size (" << httpHeaderSize 
                             << "). Dropping corrupted packet.");
                break;
            }
            
            const uint32_t pendingBefore = pendingPacket->GetSize();
            uint32_t removed = pendingPacket->RemoveHeader(header);
            if (removed != httpHeaderSize)
            {
                NS_LOG_DEBUG("Dropping corrupted server pending request: removed="
                            << removed << " expected=" << httpHeaderSize
                            << " pendingSize=" << pendingPacket->GetSize());
                pendingPacket = nullptr;
                break;
            }

            const uint32_t msgType = header.GetMessageType();
            const uint32_t videoId = header.GetVideoId();
            const uint32_t resolution = header.GetResolution();
            const uint32_t segmentId = header.GetSegmentId();
            const bool isRequest = (msgType == HTTP_REQUEST);

            if (isRequest)
            {
                const uint32_t maxSegmentId =
                    m_metadataReady ? m_metadataIndex.GetSegmentCount() : 10000;
                const bool validRequest = (videoId > 0) && (videoId <= 1024) &&
                                          IsLadderBitrate(resolution) &&
                                          (segmentId < maxSegmentId);
                if (!validRequest)
                {
                    NS_LOG_WARN("DashServer rejecting invalid HTTP_REQUEST: videoId="
                                << videoId << " resolution=" << resolution
                                << " segmentId=" << segmentId
                                << " pendingBytes=" << pendingBefore);
                    WriteMetadataTrace(videoId,
                                       segmentId,
                                       resolution,
                                       0,
                                       0,
                                       "reject_invalid");
                    pendingPacket = nullptr;
                    break;
                }
                SendSegment(videoId, resolution, segmentId, socket);
            }
        }

        if (pendingPacket && pendingPacket->GetSize() == 0)
        {
            pendingPacket = nullptr;
        }

        if (InetSocketAddress::IsMatchingType(from))
        {
            NS_LOG_INFO("At time " << Simulator::Now().GetSeconds() << "s packet sink received "
                                   << packet->GetSize() << " bytes from "
                                   << InetSocketAddress::ConvertFrom(from).GetIpv4() << " port "
                                   << InetSocketAddress::ConvertFrom(from).GetPort() << " total Rx "
                                   << m_totalRx << " bytes");
        }
        else if (Inet6SocketAddress::IsMatchingType(from))
        {
            NS_LOG_INFO("At time " << Simulator::Now().GetSeconds() << "s packet sink received "
                                   << packet->GetSize() << " bytes from "
                                   << Inet6SocketAddress::ConvertFrom(from).GetIpv6() << " port "
                                   << Inet6SocketAddress::ConvertFrom(from).GetPort()
                                   << " total Rx " << m_totalRx << " bytes");
        }
        m_rxTrace(packet, from);
    }
}

void
DashServer::HandlePeerClose(Ptr<Socket> socket)
{
    NS_LOG_FUNCTION(this << socket);
    m_pendingPackets.erase(socket);
    m_queues.erase(socket);
    m_socketList.remove(socket);
}

void
DashServer::HandlePeerError(Ptr<Socket> socket)
{
    NS_LOG_FUNCTION(this << socket);
    m_pendingPackets.erase(socket);
    m_queues.erase(socket);
    m_socketList.remove(socket);
}

void
DashServer::HandleAccept(Ptr<Socket> s, const Address& from)
{
    NS_LOG_FUNCTION(this << s << from);
    
    std::string fromInfo = "Unknown";
    if (InetSocketAddress::IsMatchingType(from))
    {
        Ipv4Address fromIp = InetSocketAddress::ConvertFrom(from).GetIpv4();
        uint16_t fromPort = InetSocketAddress::ConvertFrom(from).GetPort();
        std::ostringstream oss;
        oss << fromIp << ":" << fromPort;
        fromInfo = oss.str();
    }
    
    NS_LOG_UNCOND("DashServer - ACCEPTED connection from " << fromInfo << " at time " << Simulator::Now().GetSeconds() << "s. Total connections: " << (m_socketList.size() + 1));
        
    s->SetRecvCallback(MakeCallback(&DashServer::HandleRead, this));
    s->SetSendCallback(MakeCallback(&DashServer::DataSend, this));
    m_socketList.push_back(s);
}

void
DashServer::DataSend(Ptr<Socket> socket, uint32_t)
{
    NS_LOG_FUNCTION(this);

    // for (std::map<Ptr<Socket>, std::queue<Packet>>::iterator iter = m_queues.begin ();
    //      iter != m_queues.end (); ++iter)
    //   {
    //     HTTPHeader httpHeader;
    //     MPEGHeader mpegHeader;

    //     if (iter->second.size ())
    //       {
    //         Ptr<Packet> frame = iter->second.front ().Copy ();

    //         frame->RemoveHeader (mpegHeader);
    //         frame->RemoveHeader (httpHeader);

    //         NS_LOG_INFO ("VidId: " << httpHeader.GetVideoId ()
    //                                << " rxAv= " << iter->first->GetRxAvailable ()
    //                                << " queue= " << iter->second.size ()
    //                                << " res= " << httpHeader.GetResolution ());
    //       }
    //   }

    while (!m_queues[socket].empty())
    {
        uint32_t max_tx_size = socket->GetTxAvailable();

        if (max_tx_size <= 0)
        {
            NS_LOG_INFO("Socket Send buffer is full");
            return;
        }

        Ptr<Packet> frame = m_queues[socket].front().Copy();
        m_queues[socket].pop_front();

        uint32_t init_size = frame->GetSize();

        if (max_tx_size < init_size)
        {
            NS_LOG_INFO("Insufficient space in send buffer, fragmenting");
            Ptr<Packet> frag0 = frame->CreateFragment(0, max_tx_size);
            Ptr<Packet> frag1 = frame->CreateFragment(max_tx_size, init_size - max_tx_size);

            m_queues[socket].push_front(*frag1);
            frame = frag0;
        }

        uint32_t bytes;
        if ((bytes = socket->Send(frame)) < frame->GetSize())
        {
            NS_LOG_WARN("Couldn't send packet, though space should be available");
            Simulator::Schedule(MilliSeconds(10), &DashServer::DataSend, this, socket, 0);
        }
        else
        {
            NS_LOG_INFO("Just sent " << frame->GetSerializedSize() << " " << frame->GetSize());
            // socket->Send(Create<Packet>(0));
        }
    }
}


void
DashServer::WriteMetadataTrace(uint32_t videoId,
                               uint32_t segmentId,
                               uint32_t bitrateBps,
                               uint32_t targetBytes,
                               uint32_t totalBytes,
                               const char* source)
{
    if (m_metadataTracePath.empty())
    {
        return;
    }

    std::ofstream trace(m_metadataTracePath.c_str(), std::ios::app);
    if (!trace.is_open())
    {
        NS_LOG_WARN("DashServer could not open metadata trace file: " << m_metadataTracePath);
        return;
    }

    if (!m_metadataTraceHeaderWritten)
    {
        trace << "sim_time_s,video_id,segment_id,bitrate_bps,target_bytes,total_bytes,source\n";
        m_metadataTraceHeaderWritten = true;
    }

    trace << Simulator::Now().GetSeconds() << ","
          << videoId << ","
          << segmentId << ","
          << bitrateBps << ","
          << targetBytes << ","
          << totalBytes << ","
          << source
          << "\n";
}

void
DashServer::SendSegment(uint32_t video_id,
                        uint32_t resolution,
                        uint32_t segment_id,
                        Ptr<Socket> socket)
{
    NS_LOG_DEBUG("Generating Segment " << segment_id << " for Video " << video_id << ". Socket " << socket);
    uint32_t targetSegmentBytes = 0;
    bool metadataHit = false;
    if (m_metadataReady)
    {
        metadataHit = m_metadataIndex.GetSegmentSizeBytes(segment_id, resolution, targetSegmentBytes);
    }

    uint32_t total_segment_bytes = 0;
    int fps = 1000 / MPEG_TIME_BETWEEN_FRAMES;
    int avg_packetsize = resolution / (fps * 8);
    HTTPHeader http_header_tmp;
    MPEGHeader mpeg_header_tmp;
    Ptr<UniformRandomVariable> frame_size_gen = CreateObject<UniformRandomVariable>();
    frame_size_gen->SetAttribute("Min", DoubleValue(0));
    frame_size_gen->SetAttribute(
        "Max",
        DoubleValue(std::max(2 * avg_packetsize - (int)(mpeg_header_tmp.GetSerializedSize() +
                                                        http_header_tmp.GetSerializedSize()),
                             1)));

    for (uint32_t f_id = 0; f_id < MPEG_FRAMES_PER_SEGMENT; f_id++)
    {
        uint32_t frame_size = 0;
        if (metadataHit)
        {
            const uint32_t base = targetSegmentBytes / MPEG_FRAMES_PER_SEGMENT;
            const uint32_t rem = targetSegmentBytes % MPEG_FRAMES_PER_SEGMENT;
            frame_size = base + ((f_id < rem) ? 1u : 0u);
        }
        else
        {
            frame_size = static_cast<uint32_t>(frame_size_gen->GetValue());
        }

        HTTPHeader http_header;
        http_header.SetMessageType(HTTP_RESPONSE);
        http_header.SetVideoId(video_id);
        http_header.SetResolution(resolution);
        http_header.SetSegmentId(segment_id);

        MPEGHeader mpeg_header;
        mpeg_header.SetFrameId(f_id);
        mpeg_header.SetPlaybackTime(MilliSeconds((f_id + (segment_id * MPEG_FRAMES_PER_SEGMENT)) *
                                                 MPEG_TIME_BETWEEN_FRAMES));
        mpeg_header.SetType('B');
        mpeg_header.SetSize(frame_size);

        Ptr<Packet> frame = Create<Packet>(frame_size);
        frame->AddHeader(mpeg_header);
        frame->AddHeader(http_header);
        total_segment_bytes += frame->GetSize();

        m_queues[socket].push_back(*frame);
    }

    NS_LOG_UNCOND("SendSegment: seg=" << segment_id
                  << " rate=" << resolution
                  << " bytes=" << total_segment_bytes
                  << " expected=" << (metadataHit ? targetSegmentBytes : 0)
                  << " source=" << (metadataHit ? "metadata" : "synthetic"));
    WriteMetadataTrace(video_id,
                       segment_id,
                       resolution,
                       targetSegmentBytes,
                       total_segment_bytes,
                       metadataHit ? "metadata" : "synthetic");
    DataSend(socket, 0);
}

} // Namespace ns3
