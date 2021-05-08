# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2020-05-08

### Added
- A redesigned website hosted on GitHub Pages.

### Changed
- Improved the API response structure.

## [0.2.0] - 2020-05-03
### Added
- An initial trained model for recognizing the letters.
- Model connected to an API using Azure Web Services.
- Mock website with a rudimental drawing feature, which connects to the API and recognizes the letter.
## [0.1.0] - 2020-04-13
### Added
- Module 'nn' with various classes (listed below) to be used in the model.
- Layers: 2D Convolution layer, 2D MaxPool layer, Linear layer.
- Loss: MAE and Cross-Entropy.
- Activation: ReLU.
- A simple training script for training a model.