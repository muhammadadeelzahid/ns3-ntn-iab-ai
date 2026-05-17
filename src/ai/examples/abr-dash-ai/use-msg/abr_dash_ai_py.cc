/*
 * abr_dash_ai_py.cc
 *
 * pybind11 bindings for AbrDashCpp2PyMsg / AbrDashPy2CppMsg and Ns3AiMsgInterfaceImpl.
 *
 * Author: Muhammad Adeel Zahid <zahidma@myumanitoba.ca>
 *         University of Manitoba
 *
 * Added: pybind11 bindings for AbrDash message structs and Ns3AiMsgInterfaceImpl.
 */

#include "abr-dash-ai-msg.h"
#include "ns3-ai-msg-interface.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

using IfaceT = ns3::Ns3AiMsgInterfaceImpl<AbrDashCpp2PyMsg, AbrDashPy2CppMsg>;

PYBIND11_MODULE(ns3ai_abr_dash_ai_py, m)
{
    py::class_<AbrDashCpp2PyMsg>(m, "AbrDashCpp2PyMsg")
        .def(py::init<>())
        .def_readwrite("node_id", &AbrDashCpp2PyMsg::node_id)
        .def_readwrite("segment_id", &AbrDashCpp2PyMsg::segment_id)
        .def_readwrite("sim_time_s", &AbrDashCpp2PyMsg::sim_time_s)
        .def_readwrite("segment_fetch_time_s", &AbrDashCpp2PyMsg::segment_fetch_time_s)
        .def_readwrite("throughput_estimate_bps", &AbrDashCpp2PyMsg::throughput_estimate_bps)
        .def_readwrite("buffer_level_s", &AbrDashCpp2PyMsg::buffer_level_s)
        .def_readwrite("buffer_differential_s", &AbrDashCpp2PyMsg::buffer_differential_s)
        .def_readwrite("current_bitrate_bps", &AbrDashCpp2PyMsg::current_bitrate_bps)
        .def_readwrite("prev_bitrate_bps", &AbrDashCpp2PyMsg::prev_bitrate_bps)
        .def_readwrite("player_state", &AbrDashCpp2PyMsg::player_state)
        .def_readwrite("interruption_time_s", &AbrDashCpp2PyMsg::interruption_time_s)
        .def_readwrite("interruptions", &AbrDashCpp2PyMsg::interruptions);

    py::class_<AbrDashPy2CppMsg>(m, "AbrDashPy2CppMsg")
        .def(py::init<>())
        .def_readwrite("chosen_bitrate_bps", &AbrDashPy2CppMsg::chosen_bitrate_bps)
        .def_readwrite("delay_s", &AbrDashPy2CppMsg::delay_s);

    py::class_<IfaceT>(m, "Ns3AiMsgInterfaceImpl")
        .def(py::init<bool, bool, bool, uint32_t, const char*, const char*, const char*, const char*>())
        .def("PyRecvBegin", &IfaceT::PyRecvBegin)
        .def("PyRecvEnd", &IfaceT::PyRecvEnd)
        .def("PySendBegin", &IfaceT::PySendBegin)
        .def("PySendEnd", &IfaceT::PySendEnd)
        .def("PyGetFinished", &IfaceT::PyGetFinished)
        .def("GetCpp2PyStruct", &IfaceT::GetCpp2PyStruct, py::return_value_policy::reference)
        .def("GetPy2CppStruct", &IfaceT::GetPy2CppStruct, py::return_value_policy::reference);
}
