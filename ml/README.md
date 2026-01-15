# VisQAI: Physics-Informed Viscosity Prediction Library

**VisQAI** is a domain-specialized deep learning framework designed to predict the viscosity of high-concentration protein formulations. Unlike standard "black box" ML models, VisQAI hybridizes a Residual Neural Network with **Physics-Informed Machine Learning (PIML)** layers to respect chemical laws, ensure physical consistency, and generalize better to unseen formulations.

---

## Design Rationale

The design of this system addresses three critical challenges in biopharmaceutical viscosity modeling:

1.  **Non-Linear Concentration Effects:** Excipients (like salts or amino acids) behave differently at low vs. high concentrations. A simple linear input cannot capture the threshold where an excipient switches from "shielding" to "crowding."
    * *Solution:* **Concentration Splitting (`E_low` / `E_high`)**.
2.  **Data Scarcity in High-Viscosity Regions:** Experimental data for extremely viscous samples (>50 cP) is rare and expensive to generate, yet this is the critical failure mode we must predict.
    * *Solution:* **Physics-Informed Loss** and **Learnable Physics Priors** that enforce known behaviors (e.g., shear thinning) even where data is sparse.
3.  **Complex Protein-Excipient Interactions:** The effect of an excipient (e.g., Arginine) depends entirely on the protein's surface charge and the current interaction regime.
    * *Solution:* **Regime Gating** and **Charge-Charge Interaction (CCI)** logic.

---

## System Architecture

The system is now refactored into a modular package `src`. Below is the data flow and architectural breakdown.

### 1. The Data Processor (`src.data`)
Before entering the neural network, raw formulation data undergoes rigorous chemical feature engineering.

* **Regime Calculation:**
    The model computes a "Regime" (Near, Mixed, Far) for every sample based on the **Charge-Charge Interaction (CCI)** score.
    $$CCI = C_{Class} \times e^{-|pH - pI| / \tau}$$
    * *Rationale:* This determines if the protein is dominated by attractive or repulsive forces, which fundamentally changes how excipients work.

* **Concentration Splitting:**
    Key numeric inputs (Salts, Stabilizers, Surfactants) are split into two features based on domain-specific thresholds (e.g., Arginine @ 150mM).
    * `_low`: Captures effects up to the threshold (e.g., electrostatic shielding).
    * `_high`: Captures effects beyond the threshold (e.g., excluded volume/crowding).

### 2. The Model (`src.models`)
The model is a hybrid architecture:

* **Categorical Embeddings:** Learnable vector representations for Protein Type, Buffer Type, etc.
* **The Physics Layer (`LearnablePhysicsPrior`):**
    Instead of starting from random noise, specific weights in the network are initialized using the **Hofmeister Series** and known excipient behaviors defined in `EXCIPIENT_PRIORS`.
    * *Mechanism:* The layer outputs a correction factor based on `(Protein_Class, Regime, Excipient_Type)`.
    * *Learnability:* These priors are not frozen; the model learns a `delta` parameter to fine-tune the standard textbook values based on actual observed data.
* **Deep Residual Network:**
    The core "learning" happens in a stack of Residual Blocks (Linear -> LayerNorm -> ReLU -> Dropout) which combine the physics outputs with the standard numeric inputs.

### 3. Physics-Informed Loss (`src.loss`)
We do not train on MSE alone. The loss function includes penalty terms to enforce physical laws:

* **Shear Thinning Constraint:** Penalizes predictions where viscosity *increases* as shear rate increases (a physical impossibility for these fluids).
* **Input Gradient Constraints:** Penalizes the model if it learns physically invalid sensitivities (e.g., if adding salt *decreases* viscosity in a regime where it is known to *increase* it).

---

## Directory Structure

```text
src/
├── __init__.py          # Facade for easy imports
├── config.py            # Static Config: Thresholds, Priors (Hofmeister), Feature lists
├── data.py              # DataProcessor: Scaling, Regime calc, Splitting logic
├── models.py            # Model Arch: ResNet + LearnablePhysicsPrior + Ensemble
├── layers.py            # NN Blocks: ResidualBlock, PhysicsPrior, Adapters
├── loss.py              # PhysicsInformedLoss, Physics Masks
├── management.py        # Checkpointing, Adapter attachment, Model expansion
├── evaluation.py        # Metrics (RMSE, R2), Prediction logging
└── utils.py             # Helpers: Math, Cleaning, Tensor conversion
```
