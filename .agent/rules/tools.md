# Hardware-Accelerated Neural Network in VHDL - AI Agent Guidelines for Tools

This document outlines the tools guidelines for assisting with the development of a hardware-accelerated artificial neural network implementation in VHDL.

## Core Principles

- **Prefer OS tools when possible**:
  - When working with tasks that can be done way faster via tools like awk/sed, we prefer those tools over manual changes, for example, replacing the name of constant across all the code base.

- **When using make to simply verify that the project is running, use `make test TESTBENCH=xor`.**
