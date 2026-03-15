# fermentation

# Fermentation Data Analysis Prototype

## Overview

This project explores a simple pipeline for analyzing fermentation experiments using Python. The goal is to automatically extract key process features from time-series data, detect relevant events during the run, and generate short experiment summaries.

The project combines:

* basic data analysis
* rule-based event detection
* visualization for validation
* optional LLM-based interpretation of experimental results

It is currently a **prototype for automated bioprocess data analysis**.

---

## Features

### Feature Extraction

The script extracts summary statistics from fermentation time series, including:

* start time, end time, and experiment duration
* dissolved oxygen (initial, minimum, final, drop)
* biomass growth (initial, maximum, final, increase)
* pH dynamics (initial, min, max, final, range)
* glucose consumption

These features provide a compact representation of the fermentation run.

---

### Event Detection

Simple rule-based checks identify potential process events:

* **Low dissolved oxygen**
* **Biomass plateau**
* **Significant pH drift**

These checks help highlight possible process limitations or transitions.

---

### Visualization

Plots are generated to validate extracted features and event detection logic.
Markers highlight values such as:

* minimum DO
* maximum biomass
* pH range
* plateau checks

This allows quick visual verification of the automated analysis.

---

### Automated Run Interpretation (Optional)

The pipeline can generate a short textual interpretation of the run using a language model.
The model receives:

* experiment metadata
* extracted features
* detected events

and returns:

* a short summary
* possible explanations
* suggested next experimental steps

---

## Project Structure

```
fermentation/
│
├── prototype.py      # main analysis prototype
├── data/             # example fermentation datasets
├── reports/          # generated experiment summaries
├── memory/           # stored analysis records
└── README.md
```

---

## Requirements

Python 3.9+

Main packages:

```
pandas
matplotlib
anthropic
python-dotenv
```

Install with:

```
pip install -r requirements.txt
```

---

## Usage

Run the prototype script:

```
python prototype.py
```

The pipeline will:

1. load fermentation data
2. extract summary features
3. detect possible events
4. generate validation plots
5. optionally create a short experiment report

---

## Current Limitations

This project is an early prototype and has several limitations:

* event detection rules are simplified
* thresholds are hard-coded
* limited handling of noisy or irregular time series
* minimal validation of model-generated interpretations

Future improvements may include:

* better signal processing
* growth rate estimation
* oxygen uptake analysis
* more robust event detection
* support for multiple fermentation runs

---

## Purpose

This repository serves as an exploration of how **bioprocess experiments can be automatically summarized and interpreted using data analysis and AI tools**.

---

## Author

Marie-Batisse Heite
