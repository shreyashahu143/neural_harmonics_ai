# Neural Harmonics AI

## Setup Instructions

This project uses the external `figaro` library. Follow these steps to set it up:

### 1. Installation
1.First, clone this repository and install the Python requirements:
```bash
git clone [https://github.com/shreyashahu143/neural_harmonics_ai.git](https://github.com/shreyashahu143/neural_harmonics_ai.git)
cd neural_harmonics_ai
pip install -r requirements.txt

2. Install Figaro
This project depends on figaro but does not include it. Clone it directly into the project folder:

git clone "https://github.com/dvruette/figaro?tab=readme-ov-file"

3. Setup Checkpoints
The model checkpoints are too large for GitHub.

Download the checkpoint files from here: [https://polybox.ethz.ch/index.php/s/a0HUHzKuPPefWkW/download]

Create a folder named checkpoints inside the figaro folder.

Place the downloaded .pth or .ckpt files inside figaro/checkpoints/.