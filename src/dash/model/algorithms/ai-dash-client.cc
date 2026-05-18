/*
 * ai-dash-client.cc
 *
 * Implements AiDashClient::CalcNextSegment using ns3-ai struct-mode IPC
 * (AbrDashCpp2PyMsg / AbrDashPy2CppMsg).
 *
 * Author: Muhammad Adeel Zahid <zahidma@myumanitoba.ca>
 *         University of Manitoba
 *
 * Added: CalcNextSegment IPC with AbrDashCpp2PyMsg / AbrDashPy2CppMsg.
 */

#include "ai-dash-client.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/ai-module.h" // ns3-ai headers
#include "ns3/abr-dash-ai-msg.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("AiDashClient");

NS_OBJECT_ENSURE_REGISTERED(AiDashClient);

TypeId
AiDashClient::GetTypeId(void)
{
  static TypeId tid = TypeId("ns3::AiDashClient")
    .SetParent<DashClient>()
    .SetGroupName("Applications")
    .AddConstructor<AiDashClient>()
  ;
  return tid;
}

AiDashClient::AiDashClient()
  : m_prevBitrate(0),
    m_lastDecisionDelayS(0.0),
    m_lastThroughputEstimateBps(0.0),
    m_smoothedRttMs(-1.0),
    m_lossRateRecent(-1.0),
    m_lossEventsSinceLastDecision(0),
    m_lastDecisionTime(Seconds(0)),
    m_socketTraceConnected(false)
{
  NS_LOG_FUNCTION(this);
}

AiDashClient::~AiDashClient()
{
  NS_LOG_FUNCTION(this);
}

void
AiDashClient::UpdateRecentThroughput(double throughputBps)
{
  if (throughputBps < 0.0)
  {
    throughputBps = 0.0;
  }
  m_recentThroughputBps.push_back(throughputBps);
  while (m_recentThroughputBps.size() > kRecentHistoryWindow)
  {
    m_recentThroughputBps.pop_front();
  }
}

double
AiDashClient::ComputeThroughputEma(double alpha) const
{
  if (m_recentThroughputBps.empty())
  {
    return m_lastThroughputEstimateBps;
  }
  alpha = std::max(0.0, std::min(1.0, alpha));
  double ema = m_recentThroughputBps.front();
  for (size_t i = 1; i < m_recentThroughputBps.size(); ++i)
  {
    ema = alpha * m_recentThroughputBps[i] + (1.0 - alpha) * ema;
  }
  return ema;
}

double
AiDashClient::ComputeThroughputStd() const
{
  if (m_recentThroughputBps.size() < 2)
  {
    return 0.0;
  }
  const double sum = std::accumulate(m_recentThroughputBps.begin(), m_recentThroughputBps.end(), 0.0);
  const double mean = sum / static_cast<double>(m_recentThroughputBps.size());
  double variance = 0.0;
  for (double sample : m_recentThroughputBps)
  {
    const double diff = sample - mean;
    variance += diff * diff;
  }
  variance /= static_cast<double>(m_recentThroughputBps.size());
  return std::sqrt(std::max(0.0, variance));
}

uint32_t
AiDashClient::ComputeRecentSwitchCount() const
{
  if (m_recentMappedRates.size() < 2)
  {
    return 0;
  }
  uint32_t switches = 0;
  for (size_t i = 1; i < m_recentMappedRates.size(); ++i)
  {
    if (m_recentMappedRates[i] != m_recentMappedRates[i - 1])
    {
      ++switches;
    }
  }
  return switches;
}

void
AiDashClient::OnRttSample(Time oldValue, Time newValue)
{
  (void)oldValue;
  m_smoothedRttMs = newValue.GetMilliSeconds();
}

void
AiDashClient::OnPacketLoss(uint32_t packetNumber, uint32_t packetSize, uint8_t pathId)
{
  (void)packetNumber;
  (void)packetSize;
  (void)pathId;
  ++m_lossEventsSinceLastDecision;
}

void
AiDashClient::MaybeAttachSocketTraces()
{
  if (m_socketTraceConnected)
  {
    return;
  }

  Ptr<Socket> socket = GetSocket();
  if (!socket)
  {
    return;
  }

  bool anyConnected = false;
  anyConnected = socket->TraceConnectWithoutContext("PacketLoss",
                                                    MakeCallback(&AiDashClient::OnPacketLoss, this)) ||
                 anyConnected;
  anyConnected = socket->TraceConnectWithoutContext("RTT0",
                                                    MakeCallback(&AiDashClient::OnRttSample, this)) ||
                 anyConnected;
  anyConnected = socket->TraceConnectWithoutContext("RTT1",
                                                    MakeCallback(&AiDashClient::OnRttSample, this)) ||
                 anyConnected;
  // Some builds expose RTT (without path suffix); attempt it as well.
  anyConnected = socket->TraceConnectWithoutContext("RTT",
                                                    MakeCallback(&AiDashClient::OnRttSample, this)) ||
                 anyConnected;

  if (anyConnected)
  {
    m_socketTraceConnected = true;
    NS_LOG_INFO("AiDashClient attached QUIC RTT/loss traces for node " << GetNode()->GetId());
  }
}

void
AiDashClient::CalcNextSegment(uint32_t currRate, uint32_t& nextRate, Time& delay)
{
  NS_LOG_FUNCTION(this << currRate);
  MaybeAttachSocketTraces();

  auto* msgIface = Ns3AiMsgInterface::Get()
      ->GetInterface<AbrDashCpp2PyMsg, AbrDashPy2CppMsg>();

  if (!msgIface)
  {
      NS_LOG_WARN("Ns3AiMsgInterface not configured for AbrDashCpp2PyMsg! Using lowest rate.");
      nextRate = rates.front();
      delay = Seconds(0);
      return;
  }

  // 1. Write state to shared memory
  msgIface->CppSendBegin();
  auto* cpp2py = msgIface->GetCpp2PyStruct();
  const double throughputEstimateBps = GetBitRateEstimate();
  UpdateRecentThroughput(throughputEstimateBps);
  const double throughputEmaBps = ComputeThroughputEma();
  const double throughputStdBps = ComputeThroughputStd();
  const double segmentFetchTimeS = GetSegmentFetchTime();
  const Time now = Simulator::Now();
  const double dtSinceLastDecisionS =
      (m_lastDecisionTime > Seconds(0)) ? (now - m_lastDecisionTime).GetSeconds() : 0.0;
  if (dtSinceLastDecisionS > 0.0)
  {
    m_lossRateRecent = static_cast<double>(m_lossEventsSinceLastDecision) / dtSinceLastDecisionS;
  }
  else if (m_lossRateRecent < 0.0)
  {
    m_lossRateRecent = 0.0;
  }
  constexpr double kSegmentDurationS = 2.0; // Each metadata segment represents 2 seconds.
  cpp2py->node_id              = GetNode()->GetId();
  cpp2py->segment_id           = m_segmentId;
  cpp2py->sim_time_s           = now.GetSeconds();
  cpp2py->throughput_estimate_bps = throughputEstimateBps;
  cpp2py->buffer_level_s       = GetBufferEstimate();
  cpp2py->buffer_differential_s = GetBufferDifferential();
  cpp2py->segment_fetch_time_s = segmentFetchTimeS;
  cpp2py->current_bitrate_bps  = currRate;
  cpp2py->prev_bitrate_bps     = m_prevBitrate;
  cpp2py->player_state         = static_cast<int32_t>(m_player.m_state);
  cpp2py->interruption_time_s  = m_player.m_interruption_time.GetSeconds();
  cpp2py->interruptions        = m_player.m_interrruptions;
  cpp2py->smoothed_rtt_ms      = m_smoothedRttMs;
  cpp2py->loss_rate_recent     = m_lossRateRecent;
  cpp2py->throughput_ema_bps   = throughputEmaBps;
  cpp2py->throughput_std_bps   = throughputStdBps;
  cpp2py->download_time_ratio  = segmentFetchTimeS / kSegmentDurationS;
  cpp2py->last_decision_delay_s = m_lastDecisionDelayS;
  cpp2py->bitrate_switch_count_recent = ComputeRecentSwitchCount();
  msgIface->CppSendEnd();  // Python unblocks
  m_lossEventsSinceLastDecision = 0;
  m_lastDecisionTime = now;

  // 2. Wait for Python's decision
  msgIface->CppRecvBegin();
  auto* py2cpp = msgIface->GetPy2CppStruct();
  uint32_t chosenBitrate = py2cpp->chosen_bitrate_bps;
  double delaySeconds    = py2cpp->delay_s;
  msgIface->CppRecvEnd();

  // 3. Map chosen bitrate to the nearest valid rate
  nextRate = rates.front();
  for (uint32_t rate : rates) {
      if (chosenBitrate >= rate) {
          nextRate = rate;
      }
  }

  delay = Seconds(std::max(0.0, delaySeconds));
  m_lastDecisionDelayS = std::max(0.0, delaySeconds);
  m_prevBitrate = currRate;
  m_lastThroughputEstimateBps = throughputEstimateBps;
  m_recentMappedRates.push_back(nextRate);
  while (m_recentMappedRates.size() > kRecentHistoryWindow)
  {
    m_recentMappedRates.pop_front();
  }

  NS_LOG_INFO("AI ABR: seg=" << m_segmentId 
              << " buffer=" << cpp2py->buffer_level_s
              << " chosen=" << chosenBitrate 
              << " mapped=" << nextRate
              << " ema_tput=" << cpp2py->throughput_ema_bps
              << " tput_std=" << cpp2py->throughput_std_bps);
}

} // namespace ns3
