# Python Circuit Simulator & Schematic Parser

A Python-based electrical circuit simulator that can **analyze a circuit schematic image, convert it into a netlist, and solve the circuit automatically**.

The project combines **electrical engineering, computer vision, and numerical analysis** to turn a circuit diagram into calculated node voltages, component currents, and power dissipation.

## Features

- Reads circuit schematic images
- Uses OCR to extract component values
- Detects wires using computer vision
- Identifies resistor locations and terminals
- Determines electrical connectivity between components
- Groups connected wires into electrical nodes
- Automatically generates a SPICE-style `circuit.txt` netlist
- Solves DC resistive circuits
- Calculates:
  - Node voltages
  - Resistor currents
  - Resistor power dissipation
  - Voltage-source currents
- Supports series and parallel resistor circuits

## How It Works

The program uses the following pipeline:

```text
Circuit Image
     ↓
Image Preprocessing
     ↓
OCR Component Values
     ↓
Component Detection
     ↓
Wire Detection
     ↓
Terminal Detection
     ↓
Electrical Node Grouping
     ↓
Generate circuit.txt
     ↓
Circuit Solver
     ↓
Voltages / Currents / Power
```

## Project Structure

```text
Circuit-Solver/
│
├── main.py
├── image_parser.py
├── circuit.txt
│
└── images/
    └── test_circuit.png
```

### `image_parser.py`

Processes the circuit schematic using computer vision and OCR.

It:

1. Loads the schematic image
2. Converts the image to binary form
3. Uses EasyOCR to extract component values
4. Locates resistor symbols
5. Masks components before wire detection
6. Detects wires using the Hough Line Transform
7. Merges duplicate wire segments
8. Determines resistor terminals
9. Groups electrically connected wires
10. Assigns node numbers
11. Generates `circuit.txt`

### `main.py`

Reads the generated netlist and solves the electrical circuit.

The solver calculates:

- Node voltages
- Resistor currents
- Resistor power
- Voltage-source currents

## Example

For a circuit containing:

- 9 V voltage source
- 1 kΩ resistor
- 2.2 kΩ resistor

in series, the image parser automatically generates:

```text
V1 1 0 9.0
R1 1 2 1000.0
R2 2 0 2200.0
```

The simulator produces:

```text
===================================
       PYTHON CIRCUIT SIMULATOR
===================================

NODE VOLTAGES
-----------------------------------
Node 0 (GND): 0.000 V
Node 1: 9.000 V
Node 2: 6.188 V

R1: 2.812 mA
R2: 2.813 mA

RESISTOR POWER
-----------------------------------
R1: 7.910 mW
R2: 17.402 mW

VOLTAGE SOURCE CURRENTS
-----------------------------------
V1: -2.813 mA
```

## Parallel Circuit Support

The image parser can also identify parallel resistor connections.

For example:

```text
V1 1 0 12.0
R1 1 0 1000.0
R2 1 0 2200.0
```

Both resistors are automatically recognized as being connected between the same two electrical nodes.

## Netlist Format

The simulator currently uses the following format:

### Voltage Source

```text
V[name] [positive node] [negative node] [voltage]
```

Example:

```text
V1 1 0 9
```

### Resistor

```text
R[name] [node 1] [node 2] [resistance in ohms]
```

Example:

```text
R1 1 2 1000
```

Node `0` represents ground.

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd Circuit-Solver
```

Install the required Python packages:

```bash
pip install numpy opencv-python matplotlib easyocr
```

## Usage

Place the circuit schematic you want to analyze inside the `images` folder.

Update the image path in `image_parser.py` if necessary:

```python
IMAGE_PATH = "images/test_circuit.png"
```

Run the image parser:

```bash
python image_parser.py
```

This generates:

```text
circuit.txt
```

Then run the circuit solver:

```bash
python main.py
```

The calculated circuit parameters will be displayed in the terminal.

## Technologies

- Python
- NumPy
- OpenCV
- EasyOCR
- Matplotlib
- Modified Nodal Analysis
- Hough Line Transform
- Computer Vision
- Graph-based electrical node detection

## Current Limitations

The schematic parser is currently designed primarily for clean DC resistor circuits.

Current limitations include:

- Resistor-focused component detection
- Single DC voltage-source circuits
- Clean schematic images work best
- Component placement and labeling can affect OCR accuracy
- More complex schematic layouts may require additional topology detection
- Capacitors, inductors, diodes, and transistors are not yet supported

## Future Improvements

Planned improvements include:

- Automatic detection of additional electrical components
- Capacitor and inductor support
- AC circuit analysis
- More robust voltage-source detection
- Improved OCR error correction
- Support for arbitrary component orientation
- More complex series-parallel networks
- Automatic schematic symbol classification
- GUI for loading and solving circuit images
- Machine-learning-based component recognition

## Goal

The goal of this project is to explore how **computer vision and electrical engineering can be combined to automatically interpret and analyze circuit schematics**.

Rather than manually entering a circuit netlist, the program aims to convert:

```text
Circuit Schematic → Machine-Readable Circuit → Electrical Solution
```

## Author

Caden D'Souza
