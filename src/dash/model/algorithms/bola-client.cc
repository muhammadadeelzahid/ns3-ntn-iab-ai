/*
 * bola-client.cc
 *
 * BOLA buffer-based ABR. See bola-client.h for the reference.
 */

#include "bola-client.h"

#include "../mpeg-header.h"

#include <ns3/double.h>
#include <ns3/log.h>
#include <ns3/simulator.h>

#include <cmath>
#include <vector>

NS_LOG_COMPONENT_DEFINE("BolaClient");

namespace ns3
{
NS_OBJECT_ENSURE_REGISTERED(BolaClient);

TypeId
BolaClient::GetTypeId(void)
{
    static TypeId tid =
        TypeId("ns3::BolaClient")
            .SetParent<DashClient>()
            .AddConstructor<BolaClient>()
            .AddAttribute("BufferTargetSeconds",
                          "Target (max operating) buffer in seconds; BOLA reaches the top rung "
                          "as the buffer approaches this.",
                          DoubleValue(20.0),
                          MakeDoubleAccessor(&BolaClient::m_bufferTargetS),
                          MakeDoubleChecker<double>())
            .AddAttribute("Gp",
                          "BOLA gain parameter gp (larger => leaves the lowest rung at a lower buffer).",
                          DoubleValue(5.0),
                          MakeDoubleAccessor(&BolaClient::m_gp),
                          MakeDoubleChecker<double>());
    return tid;
}

BolaClient::BolaClient()
    : m_bufferTargetS(20.0),
      m_gp(5.0)
{
}

BolaClient::~BolaClient()
{
}

void
BolaClient::CalcNextSegment(uint32_t currRate, uint32_t& nextRate, Time& delay)
{
    // Segment playback duration in seconds (MPEG: frames/segment * ms/frame).
    const double segmentDuration =
        MPEG_FRAMES_PER_SEGMENT * MPEG_TIME_BETWEEN_FRAMES / 1000.0;

    // Current buffer occupancy in seconds.
    const double bufferLevel = GetBufferEstimate();

    // Utility = log(bitrate / lowest_bitrate), so the lowest rung has utility 0.
    const double base = static_cast<double>(rates.front());
    std::vector<double> utilities;
    utilities.reserve(rates.size());
    for (uint32_t r : rates)
    {
        utilities.push_back(std::log(static_cast<double>(r) / base));
    }

    // Vp scales the utility-vs-buffer tradeoff so the top rung is reached near bufferTarget.
    const double denom = utilities.back() + m_gp;
    const double Vp =
        (denom > 1e-9) ? (m_bufferTargetS - segmentDuration) / denom : 0.0;

    // Pick the rung maximizing the BOLA score.
    int best = 0;
    double bestScore = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < rates.size(); ++i)
    {
        const double score =
            (Vp * (utilities[i] + m_gp) - bufferLevel) / static_cast<double>(rates[i]);
        if (score > bestScore)
        {
            bestScore = score;
            best = static_cast<int>(i);
        }
    }

    nextRate = rates[best];
    delay = Seconds(0);

    NS_LOG_INFO("BOLA bufferLevel=" << bufferLevel << " Vp=" << Vp << " -> rung " << best
                                    << " (" << nextRate << " bps), currRate=" << currRate);
}

} /* namespace ns3 */
