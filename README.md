# **Lithium-ion Battery Calendar Ageing Model using Universal Differential Equations**

This repository contains the implementation of a Lithium-ion Battery Calendar Ageing Model, developed using Universal Differential Equations (UDEs). The model is based on the article titled "*Lithium-ion Battery Degradation Modelling using Universal Differential Equations: Development of a Cost-Effective Parameterisation Methodology*" by **Jishnu Ayyangatu Kuzhiyil**, **Theodoros Damoulas**, **Ferran Brosa Planella**, and **W. Dhammika Widanage**.

## **Authors and Affiliations**

- **Jishnu Ayyangatu Kuzhiyil**  
  Warwick Manufacturing Group, University of Warwick, Coventry, UK <br>
  The Faraday Institution, Quad One, Harwell Science and Innovation Campus, Didcot, UK  
  Email: [jishnu-ak.ayyangatu-kuzhiyil@warwick.ac.uk](mailto:jishnu-ak.ayyangatu-kuzhiyil@warwick.ac.uk)

- **Theodoros Damoulas**  
  Department of Computer Science and Department of Statistics, University of Warwick, Coventry, UK  
  The Alan Turing Institute, London, UK

- **Ferran Brosa Planella**  
  The Faraday Institution, Quad One, Harwell Science and Innovation Campus, Didcot, UK  
  Mathematics Institute, University of Warwick, Gibbet Hill Road, Coventry, CV4 7AL, UK

- **W. Dhammika Widanage**  
  Warwick Manufacturing Group, University of Warwick, Coventry, UK

## **Model Overview**

This repository provides a computational model for simulating the calendar ageing of Lithium-ion batteries. The model includes two approaches:

1. **Physics-Based Model**:  
   A Single Particle Model with Electrolyte (SPMe) forms the base electrochemical model, coupled with additional degradation differential equations to describe Solid Electrolyte Interphase (SEI) growth and associated pore clogging. The degradation is modeled using diffusion-limited and kinetically-limited SEI growth models along with a linear pore clogging model.

2. **UDE-Based Model**:  
   A Single Particle Model with Electrolyte (SPMe) forms the base electrochemical model, coupled with additional degradation differential equations are modeled as Universal Differential Equations, where neural networks are incorporated into the degradation differential equations to capture complex degradation behavior.

## Data Files

The project includes the following data files:

* `RPT_analysis_data.mat`: This file contains the capacity data used for model validation. .

* `RPTx_analysis_data.mat`: This file contains anode Loss of Active Material (LAM) data as used in the referenced article.
  
## **Usage Instructions**

To run the model, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/JishnuKuzhiyil/Calendar-Ageing-Modelling-using--UDE---Julia-Code.git
   cd Calendar-Ageing-Modelling-using--UDE---Julia-Code

2. **Install the required Julia packages**:
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```

3. **Open the `Main.jl` file in your Julia environment**.

4. **Modify the following variables to simulate different calendar ageing scenarios:
   - `SOC`: Set the State of Charge (e.g., `SOC = 50` for 50% charge)
   - `Temperature`: Set the ambient temperature in degrees Celsius (e.g., `Temperature = 25`)

5. **Choose the model type by setting the `Model` variable**:
   - For the physics-based model: `Model = "Physics"`
   - For the UDE model: `Model = "UDE"`

6. **Run the `Main.jl` script**:
   ```
   julia Main.jl
   ```

## Python Reproduction

A Python port of the model lives alongside the Julia sources:

* `model_parameters.py` — port of `Model_parameters.jl`
* `experiment.py` — port of `Experiment.jl` (uses `h5py` to read the v7.3 `.mat` files)
* `main.py` — port of `Main.jl`, driven by `scipy.integrate.solve_ivp` (BDF)

Install dependencies and run:

```bash
pip install -r requirements.txt
python main.py                                    # SOC=85, T=45 C, UDE
python main.py --soc 50 --temperature 25 --model Physics
python main.py --max-rpts 5                       # quick smoke test
```

### Training / fine-tuning the UDE parameters

The Python port includes three training entry points:

* `train_ude_kappa.py` keeps the neural networks fixed and refits the two
  temperature-dependent UDE scale factors (`kappa1`, `kappa2`).
* `train_ude_nn.py` fine-tunes neural-network weights with a black-box SPSA
  optimizer around the SciPy reproduction.
* `train_ude_torch.py` uses PyTorch autograd and a differentiable degradation
  ODE solver to train the neural-network weights directly.

For a robust first calibration, run the kappa trainer.  By default it uses the
paper's unweighted L2 loss for relative capacity plus LAM:

```bash
python train_ude_kappa.py --soc 85 --temperature 45 --max-nfev 30 --plot results_trained.png
python main.py --soc 85 --temperature 45 --ude-params trained_ude_T45_SOC85.npz
```

The previous uncertainty-weighted objective is still available if you want the
optimizer to give more importance to points with smaller experimental standard
deviations:

```bash
python train_ude_kappa.py --soc 85 --temperature 45 --loss std-weighted --optimizer least-squares
```

For a quicker trial while developing, fit only the first few RPT points:

```bash
python train_ude_kappa.py --soc 85 --temperature 45 --max-rpts 4 --max-nfev 15
```

If the kappa-only fit is not flexible enough, the `last-layer` mode also
fine-tunes the final dense layer of both UDE neural networks:

```bash
python train_ude_nn.py --soc 85 --temperature 45 --mode last-layer --freeze-kappa --init-params trained_ude_T45_SOC85.npz --max-nfev 80 --plot results_trained_nn.png
```

To fine-tune every neural-network parameter, use SPSA.  This is the closest
black-box equivalent to training the UDE neural networks in the Python/SciPy
port, but it is slower and noisier than the differentiable Julia/SciML training
used in the paper:

```bash
python train_ude_nn.py --soc 85 --temperature 45 --mode nn --freeze-kappa --init-params trained_ude_T45_SOC85.npz --max-nfev 120 --plot results_trained_nn.png
```

For differentiable neural-network training, install the PyTorch dependencies
and run:

```bash
python train_ude_torch.py --soc 85 --temperature 45 --init-params trained_ude_T45_SOC85.npz --epochs 300 --validate-main --plot results_trained_torch.png
```

The PyTorch-native `nn.Module` implementation is available as:

```bash
python train_ude_torch_module.py --soc 85 --temperature 45 --init-params trained_ude_T45_SOC85.npz --epochs 300 --validate-main --plot results_trained_torch_module.png
```

`train_ude_torch.py` trains a differentiable calendar-ageing degradation
subsystem and exports parameters in the same `.npz` format consumed by
`main.py`.  The optional `--validate-main` step evaluates the exported
parameters in the full SciPy SPMe/RPT reproduction.

The SciPy/SPSA trainer is slower than a forward simulation because each
optimizer evaluation runs the full calendar-ageing experiment through the stiff
ODE solver.  The PyTorch trainer is faster per update because gradients come
from autograd through the differentiable degradation ODE.

The Julia driver uses a singular mass-matrix DAE for the CV current-hold
algebraic constraint. SciPy's `solve_ivp` does not support DAEs natively, so
the Python port enforces that constraint with a stiff penalty ODE handled by
the BDF integrator. End-to-end results match the experimental capacity / LAM
trends; small differences vs. Julia are expected from the change of integrator.

## Output

Plots showing capacity and anode LAM measurements and corresponding model predictions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) [2024] [Jishnu Ayyangatu Kuzhiyil]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## How to Cite

If you use this software in your research, please cite our article:

```
Jishnu Ayyangatu Kuzhiyil, Theodoros Damoulas, Ferran Brosa Planella, W. Dhammika Widanage,
Lithium-ion battery degradation modelling using universal differential equations: Development of a cost-effective parameterisation methodology,
Applied Energy, Volume 382,2025,125221,ISSN 0306-2619,
https://doi.org/10.1016/j.apenergy.2024.125221.
```


