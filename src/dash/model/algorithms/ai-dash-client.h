#ifndef AI_DASH_CLIENT_H
#define AI_DASH_CLIENT_H

#include "ns3/dash-client.h"

namespace ns3 {

/**
 * \ingroup dash
 * \brief ABR algorithm that uses an external AI/LLM model via ns3-ai shared memory.
 * 
 * This client blocks simulation execution during CalcNextSegment to wait for a 
 * bitrate decision from the connected Python process.
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

  uint32_t m_prevBitrate;
};

} // namespace ns3

#endif /* AI_DASH_CLIENT_H */
