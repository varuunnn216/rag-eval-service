# Nimbus Bike Share Operations & Rider Guide

## Pricing Tiers, Unlock Fees, and 2026 Tariff Revision

Under the legacy 2024 launch tariff, Casual Pass riders paid a ₹15 unlock fee and ₹2.50 per minute after the first 10 minutes. **Effective January 1, 2026 (Superseding Tariff v3.2):** the Casual Pass unlock fee was eliminated (₹0 unlock), but per-minute billing now begins immediately at ₹3.50 per minute for pedal bikes and ₹6.00 per minute for Nimbus Volt e-bikes.

| Plan Tier | Monthly Fee | Included Free Ride Window | E-Bike Per-Minute Rate | Max Daily Cap |
| :--- | :--- | :--- | :--- | :--- |
| Casual Pass | ₹0 | 0 minutes | ₹6.00 / min | ₹480 |
| Commuter Plus | ₹499 | First 30 minutes per trip | ₹2.00 / min after 30m | ₹250 |
| Campus Annual | ₹399 / mo | First 45 minutes per trip | ₹1.50 / min after 45m | ₹180 |

Riders who dock a Nimbus Volt e-bike at an designated uphill "Boost Hub" receive an instant ₹25 ride credit applied to their wallet balance.

## Docking Station Telemetry and Firmware CLI

Each solar-powered docking bollard runs an embedded controller that communicates with the central fleet broker over MQTT v5. When a solenoid latch jams after a brownout, field technicians must run the diagnostic reset command over the local maintenance serial port:

```bash
nimbus-ctl dock --bay-id=4 --reset-latch --solenoid-pulse-ms=4500 --verify-hall
```

If the Hall-effect sensor still reports fault code `E_LATCH_STUCK_09` after two pulses, the bay is automatically quarantined in the backend (`status = MAINTENANCE_LOCK`) and excluded from rider app availability counts.

## Battery Swapping and Cold-Weather Range Derating

Nimbus Volt e-bikes use removable 48V 14.4Ah lithium iron phosphate (LFP) battery packs housed inside the downtube. When a pack's state of charge (SoC) drops below 18%, the bike's cellular modem dispatches a swap ticket to the nearest field van and restricts motor assist to Eco mode (15 km/h top assist speed). At or below 5% SoC, the motor disengages completely while reserving 12V power for the LED headlamp and electronic rear wheel lock for up to 36 hours.

During winter operations, regenerative braking is disabled whenever the internal pack thermistor reads below -4°C to prevent lithium plating on the anode. If ambient pack temperature reaches -12°C, the battery management system (BMS) opens the main discharge contactor altogether until the pack warms above -8°C.

## Geofence Penalties, Improper Parking, and Lost Bike Liability

All trips must end inside a physical Nimbus docking bay or a painted virtual corral marked on the rider map. Ending a trip outside an approved corral triggers a ₹150 improper parking surcharge. If a rider leaves a bike outside the municipal service boundary (the red geofence zone), the rear hub lock engages once the bike comes to a complete stop and a ₹1,200 field-retrieval fee is billed after 4 hours.

If a checked-out bike is not returned within 48 consecutive hours and telemetry pings cease, the trip is classified as an unrecovered loss and the rider's payment method on file is charged the replacement liability cap of ₹18,500 (or ₹42,000 for a Nimbus Volt e-bike), unless a police theft report number is submitted within 72 hours.
