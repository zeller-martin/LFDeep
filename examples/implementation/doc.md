This particular implementation uses a Neuralynx data acquisition system (LabLynx). Data can be streamed from the device via Neuralynx Netcom API, which is most conveniently provided in Matlab. Therefore we use a Python wrapper around the Matlab API and call the Netcom functions via the Matlab engine.

An ARM-Cortex-M7 based microcontroller (Teensy 4.1) receives extracted phase information with LabLynx from the PC via Serial, and synchronizes its own clock with the clock on the LabLynx. This enables real-time stimulation conditional on the estimated phase.
