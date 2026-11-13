/*
 * bola-client.h
 *
 * BOLA: Near-Optimal Bitrate Adaptation for Online Videos (Spiteri, Urgaonkar,
 * Sitaraman, IEEE/ACM ToN 2020 / INFOCOM 2016). Buffer-based ABR derived from
 * Lyapunov optimization: per segment pick the rung maximizing
 *   (Vp*(utility[i] + gp) - bufferLevel) / bitrate[i],
 * with utility[i] = log(bitrate[i] / bitrate[0]) and
 *   Vp = (bufferTarget - segmentDuration) / (utility[top] + gp).
 * Reference: sabre (UMass-LIDS) and dash.js BolaRule.
 */

#ifndef BOLA_CLIENT_H_
#define BOLA_CLIENT_H_

#include <ns3/dash-client.h>

namespace ns3
{

class BolaClient : public DashClient
{
  public:
    static TypeId GetTypeId(void);

    BolaClient();
    virtual ~BolaClient();

    virtual void CalcNextSegment(uint32_t currRate, uint32_t& nextRate, Time& delay);

  private:
    double m_bufferTargetS; // target (max operating) buffer in seconds -> sets where BOLA reaches top rung
    double m_gp;            // BOLA gain parameter (controls how eagerly it leaves the lowest rung)
};

} /* namespace ns3 */

#endif /* BOLA_CLIENT_H_ */
