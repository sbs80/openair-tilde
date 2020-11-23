# openair-tilde
A convolution reverb external for Pure Data (Pd) with the capability to switch impulse responses on the fly. This Pd external is designed to be used in interactive immersive simulation applications such as virtual reality or gaming.

When the user sends an openair~ external object a request to swich impulse responses, it responds by smoothly cross-fading between the old and new convolution streams.

## How to compile the external run the demo app in Linux

1. Clone this repository and cd to its main directory.
2. Build the openair~ PD external by typing: `make`
3. Run the demo app by typing: `pd -lib Gem openair_demo.pd`

Note: To run the demo, PD with the GEM library must be installed. However the openair external itself does not require the GEM library to be installed.
