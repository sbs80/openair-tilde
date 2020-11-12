# openair-tilde
A convolution reverb external for Pure Data (Pd) with the capability to switch impulse responses on the fly. This Pd external is designed to be used in interactive immersive simulation applications such as virutal reality or gaming.

When the user sends an openair~ external object a request to swich impulse responses, it responds by smoothly cross-fading between the old and new convolution streams.

## How to compile the external run the demo app in Linux

1. Make sure PD with the GEM library is installed.
2. Clone this repository and cd to its main directory.
3. Build the openair~ PD external by typing `make`.
4. Run the demo app by typing `pd -lib Gem openair_demo.pd`.
