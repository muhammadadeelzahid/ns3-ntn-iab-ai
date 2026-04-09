/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include "quic-socket-state-registry.h"

#include "quic-socket-base.h" // QuicSocketState

#include "ns3/log.h"

#include <algorithm>
#include <set>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("QuicSocketStateRegistry");

static std::set<QuicSocketState*> g_activeTcbs;

void
QuicSocketStateRegistry::Register(QuicSocketState* tcb)
{
  if (tcb == nullptr)
    {
      NS_LOG_DEBUG("Register skipped: null tcb");
      return;
    }
  const bool inserted = g_activeTcbs.insert(tcb).second;
  NS_LOG_DEBUG("Register tcb=" << static_cast<void*>(tcb) << " inserted=" << inserted
                               << " active_count=" << g_activeTcbs.size());
}

void
QuicSocketStateRegistry::Unregister(QuicSocketState* tcb)
{
  if (tcb == nullptr)
    {
      NS_LOG_DEBUG("Unregister skipped: null tcb");
      return;
    }
  auto it = g_activeTcbs.find(tcb);
  if (it != g_activeTcbs.end())
    {
      g_activeTcbs.erase(it);
      NS_LOG_DEBUG("Unregister tcb=" << static_cast<void*>(tcb) << " removed=true"
                                     << " active_count=" << g_activeTcbs.size());
    }
  else
    {
      NS_LOG_DEBUG("Unregister tcb=" << static_cast<void*>(tcb)
                                     << " removed=false (not in registry) active_count="
                                     << g_activeTcbs.size());
    }
}

std::vector<QuicSocketState*>
QuicSocketStateRegistry::GetActiveTcbs()
{
  const std::size_t n = g_activeTcbs.size();
  NS_LOG_DEBUG("GetActiveTcbs snapshot size=" << n);
  return std::vector<QuicSocketState*>(g_activeTcbs.begin(), g_activeTcbs.end());
}

std::size_t
QuicSocketStateRegistry::Size()
{
  const std::size_t n = g_activeTcbs.size();
  NS_LOG_DEBUG("Size active_count=" << n);
  return n;
}

} // namespace ns3

