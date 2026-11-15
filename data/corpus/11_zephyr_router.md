# Zephyr ZX-400 Router Guide

## Overview

The Zephyr ZX-400 is a home Wi-Fi router that costs 89 USD. It supports up to 64 connected devices at the same time. The box contains the router, a 12V 2A power adapter, a 1 m Ethernet cable and a quick start card.

## Initial Setup

1. Plug in the power adapter and wait until the green LED blinks.
2. Connect the modem to the WAN port.
   - The WAN port is the blue port.
   - The four yellow ports are LAN ports for your own devices.
3. Open the Zephyr app and scan the QR code on the bottom of the router.
4. Choose a Wi-Fi name and a password of at least 10 characters.

The router admin page is at 192.168.50.1. The default admin password is printed on the label under the router.

## LED Indicators

| LED            | Meaning                              | What to do                       |
|----------------|--------------------------------------|----------------------------------|
| Solid green    | Online and working                   | Nothing                          |
| Blinking green | Starting up                          | Wait up to 2 minutes             |
| Solid red      | No internet connection               | Check the modem cable, then restart the router |
| Blinking red   | Hardware fault                       | Contact support                  |
| Solid blue     | WPS pairing is active for 2 minutes |
| Off            | No power                             | Check the power adapter          |

## Wi-Fi Bands

| Band    | Maximum speed | Range  | Best for                        |
|---------|---------------|--------|---------------------------------|
| 2.4 GHz | 300 Mbps      | Long   | Smart home devices and sensors  |
| 5 GHz   | 1200 Mbps     | Medium | Streaming and gaming            |

## Error Codes

- E-101: The WAN cable is disconnected. Check both ends of the cable.
- E-204: The DNS server is not responding. Set the DNS server to 1.1.1.1 in the admin page.
- E-310: The firmware update failed. Wait 5 minutes and try again. Do not unplug the router during an update.
- E-407: The router is overheating. Move it to a ventilated place. The router shuts down automatically at 70 degrees Celsius.

## Factory Reset

To reset the router, hold the Reset button for 10 seconds while the router is powered on, until all LEDs flash once. A factory reset erases all settings, including the Wi-Fi name and password. A short press of 3 seconds only restarts the router and keeps all settings. You can also reset from the app under Settings, System, Reset.

## Firmware History

| Version | Date          | Changes                                      |
|---------|---------------|----------------------------------------------|
| 1.0.2   | March 2025    | Initial release                              |
| 1.1.0   | August 2025   | Added guest Wi-Fi                            |
| 1.2.0   | January 2026  | Added parental controls and fixed the E-204 bug |
| 1.2.1   | April 2026    | Security patch                               |

The latest firmware version is 1.2.1.

## Warranty and Returns

The warranty lasts 24 months from the purchase date and covers hardware defects. It does not cover water damage or damage caused by a non-original power adapter. An unopened router can be returned within 30 days for a full refund. An opened router can be returned within 15 days, with a restocking fee of 10 percent of the price.

## Support

Phone support is not available. Support is provided by email and in-app chat only. Chat is open from 8:00 AM to 8:00 PM UTC. Email replies usually arrive within 2 business days.
