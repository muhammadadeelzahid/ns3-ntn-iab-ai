/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Registry of active QUIC socket state (TCB) pointers.
 *
 * Purpose:
 * - Provide robust discovery of active connections/subflows without relying on Config paths,
 *   which can be brittle when sockets are created late.
 *
 * Notes:
 * - The registry stores raw pointers; it does not participate in reference counting.
 * - Registration/unregistration must follow the lifetime of the owning object (MpQuicSubFlow).
 * - ns-3 runs single-threaded, so no locking is used.
 */

#ifndef QUIC_SOCKET_STATE_REGISTRY_H
#define QUIC_SOCKET_STATE_REGISTRY_H

#include <cstddef>
#include <vector>

namespace ns3 {

class QuicSocketState;

class QuicSocketStateRegistry
{
public:
  static void Register(QuicSocketState* tcb);
  static void Unregister(QuicSocketState* tcb);

  /** Return a snapshot of active TCB pointers. */
  static std::vector<QuicSocketState*> GetActiveTcbs();

  static std::size_t Size();
};

} // namespace ns3

#endif // QUIC_SOCKET_STATE_REGISTRY_H

