#include "ai-dash-client.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/ai-module.h" // ns3-ai headers
#include "ns3/abr-dash-ai-msg.h"
#include <algorithm>

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
  : m_prevBitrate(0)
{
  NS_LOG_FUNCTION(this);
}

AiDashClient::~AiDashClient()
{
  NS_LOG_FUNCTION(this);
}

void
AiDashClient::CalcNextSegment(uint32_t currRate, uint32_t& nextRate, Time& delay)
{
  NS_LOG_FUNCTION(this << currRate);

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
  cpp2py->node_id              = GetNode()->GetId();
  cpp2py->segment_id           = m_segmentId;
  cpp2py->sim_time_s           = Simulator::Now().GetSeconds();
  cpp2py->throughput_estimate_bps = GetBitRateEstimate();
  cpp2py->buffer_level_s       = GetBufferEstimate();
  cpp2py->buffer_differential_s = GetBufferDifferential();
  cpp2py->segment_fetch_time_s = GetSegmentFetchTime();
  cpp2py->current_bitrate_bps  = currRate;
  cpp2py->prev_bitrate_bps     = m_prevBitrate;
  cpp2py->player_state         = static_cast<int32_t>(m_player.m_state);
  cpp2py->interruption_time_s  = m_player.m_interruption_time.GetSeconds();
  cpp2py->interruptions        = m_player.m_interrruptions;
  msgIface->CppSendEnd();  // Python unblocks

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
  m_prevBitrate = currRate;

  NS_LOG_INFO("AI ABR: seg=" << m_segmentId 
              << " buffer=" << cpp2py->buffer_level_s
              << " chosen=" << chosenBitrate 
              << " mapped=" << nextRate);
}

} // namespace ns3
