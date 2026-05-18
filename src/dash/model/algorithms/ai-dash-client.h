/*
 * ai-dash-client.h
 *
 * Application-layer DASH ABR client that exchanges state and bitrate decisions
 * with an external Python process via ns3-ai shared-memory IPC.
 *
 * Author: Muhammad Adeel Zahid <zahidma@myumanitoba.ca>
 *         University of Manitoba
 *
 * Added: AiDashClient — application-layer ABR via ns3-ai shared memory.
 */

#ifndef AI_DASH_CLIENT_H
#define AI_DASH_CLIENT_H

#include "ns3/dash-client.h"
#include <deque>

namespace ns3 {

/**
 * \ingroup dash
 * \brief ABR algorithm that uses an external AI/LLM model via ns3-ai shared memory.
 *
 * On each segment request, publishes AbrDashCpp2PyMsg to Python and blocks until
 * AbrDashPy2CppMsg (chosen bitrate and optional delay) is received.
 */
class AiDashClient : public DashClient
{
public:
  static TypeId GetTypeId(void);
  AiDashClient();
  virtual ~AiDashClient();

private:
  /**
   * Overrides DashClient::CalcNextSegment
   * Uses Ns3AiMsgInterface to send state to Python and block until an action is received.
   */
  virtual void CalcNextSegment(uint32_t currRate, uint32_t& nextRate, Time& delay) override;

  void UpdateRecentThroughput(double throughputBps);
  double ComputeThroughputEma(double alpha = 0.4) const;
  double ComputeThroughputStd() const;
  uint32_t ComputeRecentSwitchCount() const;
  void MaybeAttachSocketTraces();
  void OnRttSample(Time oldValue, Time newValue);
  void OnPacketLoss(uint32_t packetNumber, uint32_t packetSize, uint8_t pathId);

  uint32_t m_prevBitrate;
  double m_lastDecisionDelayS;
  double m_lastThroughputEstimateBps;
  double m_smoothedRttMs;
  double m_lossRateRecent;
  uint32_t m_lossEventsSinceLastDecision;
  Time m_lastDecisionTime;
  bool m_socketTraceConnected;
  std::deque<double> m_recentThroughputBps;
  std::deque<uint32_t> m_recentMappedRates;
  static constexpr uint32_t kRecentHistoryWindow = 8;
};

} // namespace ns3

#endif /* AI_DASH_CLIENT_H */
