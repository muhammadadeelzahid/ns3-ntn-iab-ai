/*
 * pybind11 bindings for QUIC BBR AI msg-interface structs.
 *
 * This follows the same pattern as src/ai/examples/a-plus-b/use-msg-vec/apb_py.cc
 * so Python can access:
 * - the Cpp2Py/Py2Cpp structs
 * - the ns3::Ns3AiMsgInterfaceImpl<Cpp2Py, Py2Cpp> class
 * - vectors when use_vector=true
 */

#include "quic-bbr-ai-msg.h"

#include "ns3-ai-msg-interface.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

using IfaceT = ns3::Ns3AiMsgInterfaceImpl<QuicBbrAiCpp2PyMsg, QuicBbrAiPy2CppMsg>;

PYBIND11_MAKE_OPAQUE(IfaceT::Cpp2PyMsgVector);
PYBIND11_MAKE_OPAQUE(IfaceT::Py2CppMsgVector);

PYBIND11_MODULE(ns3ai_quic_bbr_ai_msg_py, m)
{
    py::class_<QuicBbrAiConnObs>(m, "QuicBbrAiConnObs")
        .def(py::init<>())
        .def_readwrite("node_id", &QuicBbrAiConnObs::node_id)
        .def_readwrite("conn_id", &QuicBbrAiConnObs::conn_id)
        .def_readwrite("path_id", &QuicBbrAiConnObs::path_id)
        .def_readwrite("quic_connection_id", &QuicBbrAiConnObs::quic_connection_id)
        .def_readwrite("time_s", &QuicBbrAiConnObs::time_s)
        .def_readwrite("dt_s", &QuicBbrAiConnObs::dt_s)
        .def_readwrite("bytes_in_flight_bytes", &QuicBbrAiConnObs::bytes_in_flight_bytes)
        .def_readwrite("unacked_data_bytes", &QuicBbrAiConnObs::unacked_data_bytes)
        .def_readwrite("smoothed_rtt_s", &QuicBbrAiConnObs::smoothed_rtt_s)
        .def_readwrite("rttvar_s", &QuicBbrAiConnObs::rttvar_s)
        .def_readwrite("latest_rtt_s", &QuicBbrAiConnObs::latest_rtt_s)
        .def_readwrite("min_rtt_s", &QuicBbrAiConnObs::min_rtt_s)
        .def_readwrite("acked_bytes_since_last", &QuicBbrAiConnObs::acked_bytes_since_last)
        .def_readwrite("acked_bytes_over_dt", &QuicBbrAiConnObs::acked_bytes_over_dt)
        .def_readwrite("delivered_bytes_over_dt", &QuicBbrAiConnObs::delivered_bytes_over_dt)
        .def_readwrite("throughput_bps_over_dt", &QuicBbrAiConnObs::throughput_bps_over_dt)
        .def_readwrite("delivery_rate_bps", &QuicBbrAiConnObs::delivery_rate_bps)
        .def_readwrite("ack_interarrival_last_s", &QuicBbrAiConnObs::ack_interarrival_last_s)
        .def_readwrite("ack_interarrival_mean_s_over_dt", &QuicBbrAiConnObs::ack_interarrival_mean_s_over_dt)
        .def_readwrite("ack_interarrival_min_s_over_dt", &QuicBbrAiConnObs::ack_interarrival_min_s_over_dt)
        .def_readwrite("ack_interarrival_max_s_over_dt", &QuicBbrAiConnObs::ack_interarrival_max_s_over_dt)
        .def_readwrite("loss_event_count_over_dt", &QuicBbrAiConnObs::loss_event_count_over_dt)
        .def_readwrite("bytes_lost_over_dt", &QuicBbrAiConnObs::bytes_lost_over_dt)
        .def_readwrite("packet_loss_since_last", &QuicBbrAiConnObs::packet_loss_since_last)
        .def_readwrite("reordering_event_count_over_dt", &QuicBbrAiConnObs::reordering_event_count_over_dt)
        .def_readwrite("dup_ack_count_over_dt", &QuicBbrAiConnObs::dup_ack_count_over_dt)
        .def_readwrite("btlbw_bps", &QuicBbrAiConnObs::btlbw_bps)
        .def_readwrite("rtprop_s", &QuicBbrAiConnObs::rtprop_s)
        .def_readwrite("pacing_gain", &QuicBbrAiConnObs::pacing_gain)
        .def_readwrite("cwnd_gain", &QuicBbrAiConnObs::cwnd_gain)
        .def_readwrite("pacing_rate_bps", &QuicBbrAiConnObs::pacing_rate_bps)
        .def_readwrite("cwnd_bytes", &QuicBbrAiConnObs::cwnd_bytes)
        .def_readwrite("target_cwnd_bytes", &QuicBbrAiConnObs::target_cwnd_bytes)
        .def_readwrite("bbr_mode", &QuicBbrAiConnObs::bbr_mode)
        .def_readwrite("is_app_limited", &QuicBbrAiConnObs::is_app_limited);

    py::class_<QuicBbrAiCpp2PyMsg>(m, "QuicBbrAiCpp2PyMsg")
        .def(py::init<>())
        .def_readwrite("obs", &QuicBbrAiCpp2PyMsg::obs);

    py::class_<QuicBbrAiPy2CppMsg>(m, "QuicBbrAiPy2CppMsg")
        .def(py::init<>())
        .def_readwrite("node_id", &QuicBbrAiPy2CppMsg::node_id)
        .def_readwrite("conn_id", &QuicBbrAiPy2CppMsg::conn_id)
        .def_readwrite("path_id", &QuicBbrAiPy2CppMsg::path_id)
        .def_readwrite("quic_connection_id", &QuicBbrAiPy2CppMsg::quic_connection_id)
        .def_readwrite("alpha", &QuicBbrAiPy2CppMsg::alpha)
        .def_readwrite("has_target_pacing_gain", &QuicBbrAiPy2CppMsg::has_target_pacing_gain)
        .def_readwrite("target_pacing_gain", &QuicBbrAiPy2CppMsg::target_pacing_gain)
        .def_readwrite("max_step_pacing_gain", &QuicBbrAiPy2CppMsg::max_step_pacing_gain)
        .def_readwrite("has_target_cwnd_gain", &QuicBbrAiPy2CppMsg::has_target_cwnd_gain)
        .def_readwrite("target_cwnd_gain", &QuicBbrAiPy2CppMsg::target_cwnd_gain)
        .def_readwrite("max_step_cwnd_gain", &QuicBbrAiPy2CppMsg::max_step_cwnd_gain)
        .def_readwrite("has_pacing_rate_cap_bps", &QuicBbrAiPy2CppMsg::has_pacing_rate_cap_bps)
        .def_readwrite("pacing_rate_cap_bps", &QuicBbrAiPy2CppMsg::pacing_rate_cap_bps)
        .def_readwrite("has_cwnd_cap_bytes", &QuicBbrAiPy2CppMsg::has_cwnd_cap_bytes)
        .def_readwrite("cwnd_cap_bytes", &QuicBbrAiPy2CppMsg::cwnd_cap_bytes)
        .def_readwrite("has_reordering_threshold", &QuicBbrAiPy2CppMsg::has_reordering_threshold)
        .def_readwrite("reordering_threshold", &QuicBbrAiPy2CppMsg::reordering_threshold)
        .def_readwrite("max_step_reordering_threshold",
                       &QuicBbrAiPy2CppMsg::max_step_reordering_threshold)
        .def_readwrite("has_time_reordering_fraction", &QuicBbrAiPy2CppMsg::has_time_reordering_fraction)
        .def_readwrite("time_reordering_fraction", &QuicBbrAiPy2CppMsg::time_reordering_fraction)
        .def_readwrite("max_step_time_reordering_fraction",
                       &QuicBbrAiPy2CppMsg::max_step_time_reordering_fraction);

    py::class_<IfaceT::Cpp2PyMsgVector>(m, "QuicBbrAiCpp2PyVector")
        .def(
            "resize",
            static_cast<void (IfaceT::Cpp2PyMsgVector::*)(IfaceT::Cpp2PyMsgVector::size_type)>(
                &IfaceT::Cpp2PyMsgVector::resize))
        .def("__len__", &IfaceT::Cpp2PyMsgVector::size)
        .def(
            "__getitem__",
            [](IfaceT::Cpp2PyMsgVector& vec, uint32_t i) -> QuicBbrAiCpp2PyMsg& {
                if (i >= vec.size())
                {
                    throw std::out_of_range("QuicBbrAiCpp2PyVector index out of range");
                }
                return vec.at(i);
            },
            py::return_value_policy::reference);

    py::class_<IfaceT::Py2CppMsgVector>(m, "QuicBbrAiPy2CppVector")
        .def(
            "resize",
            static_cast<void (IfaceT::Py2CppMsgVector::*)(IfaceT::Py2CppMsgVector::size_type)>(
                &IfaceT::Py2CppMsgVector::resize))
        .def("__len__", &IfaceT::Py2CppMsgVector::size)
        .def(
            "__getitem__",
            [](IfaceT::Py2CppMsgVector& vec, uint32_t i) -> QuicBbrAiPy2CppMsg& {
                if (i >= vec.size())
                {
                    throw std::out_of_range("QuicBbrAiPy2CppVector index out of range");
                }
                return vec.at(i);
            },
            py::return_value_policy::reference);

    py::class_<IfaceT>(m, "Ns3AiMsgInterfaceImpl")
        .def(py::init<bool, bool, bool, uint32_t, const char*, const char*, const char*, const char*>())
        .def("PyRecvBegin", &IfaceT::PyRecvBegin)
        .def("PyRecvEnd", &IfaceT::PyRecvEnd)
        .def("PySendBegin", &IfaceT::PySendBegin)
        .def("PySendEnd", &IfaceT::PySendEnd)
        .def("PyGetFinished", &IfaceT::PyGetFinished)
        .def("GetCpp2PyStruct", &IfaceT::GetCpp2PyStruct, py::return_value_policy::reference)
        .def("GetPy2CppStruct", &IfaceT::GetPy2CppStruct, py::return_value_policy::reference)
        .def("GetCpp2PyVector", &IfaceT::GetCpp2PyVector, py::return_value_policy::reference)
        .def("GetPy2CppVector", &IfaceT::GetPy2CppVector, py::return_value_policy::reference);
}

