SYNTHETIC DEMO DATA - written for the SkillLeak AI demo. Not a real university lab manual.

# Lab Manual: EE-357L Computer Communication Networks Lab

General instructions: Each lab is completed in Cisco Packet Tracer unless stated otherwise. Students follow the step-by-step procedure, record the outputs shown, and submit a lab report containing screenshots and answers to the post-lab questions.

## Lab 1: Introduction to Cisco Packet Tracer and Network Cabling
Objective: Become familiar with the Packet Tracer interface and connect PCs, switches and routers using the correct cable types.
Procedure: Place two PCs and a switch, connect them with straight-through cables, assign the given IP addresses and verify connectivity with ping.
Post-lab: Name the cable types used and state when a crossover cable is needed.

## Lab 2: IPv4 Addressing and Subnetting
Objective: Apply subnetting to divide a given network into subnets.
Procedure: Divide the network 192.168.10.0/24 into four equal subnets. Assign addresses to the PCs in each subnet and verify connectivity within each subnet using ping.
Post-lab: Calculate the number of usable hosts per subnet.

## Lab 3: Static Routing
Objective: Configure static routes between three routers.
Procedure: Build the given topology of three routers. Configure the interface IP addresses exactly as listed in Table 3.1. Enter the static routes provided in the procedure and verify end-to-end connectivity using ping and the show ip route command.
Post-lab: Explain the meaning of each field in the routing table.

## Lab 4: VLAN Configuration and Trunking
Objective: Configure VLANs on a switch and a trunk link between two switches.
Procedure: Create VLAN 10 and VLAN 20, assign the listed access ports, configure the trunk port and verify with show vlan brief.
Post-lab: Explain why PCs in different VLANs cannot communicate without a router.

## Lab 5: Single-Area OSPF
Objective: Configure OSPF on three routers.
Procedure: Enable OSPF process 1 on all routers, advertise the given networks in area 0, and verify neighbour adjacencies using show ip ospf neighbor.
Post-lab: Compare the OSPF routing table with the static routing table from Lab 3.

## Lab 6: Observing TCP with Wireshark
Objective: Observe the TCP three-way handshake.
Procedure: Start a Wireshark capture, open a web page in the browser, stop the capture, apply the filter tcp, and take a screenshot of the SYN, SYN-ACK and ACK packets.
Post-lab: Write down the sequence and acknowledgement numbers shown in the screenshot.

## Lab 7: DHCP and DNS Services
Objective: Configure a DHCP server and a DNS server in Packet Tracer.
Procedure: Configure the DHCP pool on the server with the parameters given, set PCs to obtain addresses automatically, add the DNS record for www.sample.local and verify name resolution from a PC.
Post-lab: Describe the four messages of the DHCP process.

## Lab 8: Standard and Extended Access Control Lists
Objective: Configure ACLs to filter traffic.
Procedure: Apply the given extended ACL on the router interface to block HTTP traffic from the 192.168.20.0/24 network and verify that the traffic is blocked.
Post-lab: State the difference between standard and extended ACLs.
