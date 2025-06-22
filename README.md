# Neural-Network-for-arc

This repository contains the implementation for the Master's thesis:

"A Context-Aware Neural Network for the Abstraction and Reasoning Corpus (ARC)"

This project introduces a novel two-stage neural architecture designed to solve abstract reasoning tasks from the ARC dataset. Unlike traditional approaches that rely on hand-crafted rules or program synthesis, this work explores whether a purely neural system can implicitly learn reasoning patterns from a few examples.

🧠 Architecture

The model consists of two sequential components:

    Vision Transformer
    Analyzes input/output demonstration grids to infer a puzzle’s logic and compress it into a context vector.

    U-Net Decoder
    Conditioned on the context vector using cross-attention, this module applies the inferred logic to transform a new, unseen input grid.

    A new model is trained from scratch for each puzzle, ensuring it learns the logic uniquely from the provided demonstrations.

🚀 Usage

    Select Puzzles
    Modify the PUZZLE_ID list in main.py to specify which puzzles to run.

    Run the Model

    python main.py

    Output

        Visualizations will be saved in the visualizations/ folder.

        Metrics will be stored in results.json.

📖 Citation

If you use this work in your research, please cite the original thesis:

@mastersthesis{Abdelhamed2025ARC,
  author  = {Ahmed Abdelhamed},
  title   = {A Context-Aware Neural Network for the Abstraction and Reasoning Corpus (ARC)},
  school  = {African Institute for Mathematical Sciences (AIMS)},
  year    = {2025},
  note    = {Supervised by Prof. Ulrich Paquet and Jaron Cohen}
}