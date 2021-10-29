# Fully differentiable optimization protocols for non-equilibrium steady states
**authors:** Rodrigo A. Vargas-Hernandez, Ricky T. Q. Chen, Kenneth A. Jung, and Paul Brumer


---------------------------------------
**GOAL:**

Optimizing quantum non-equilibrium steady state systems using gradient-based methods power by automatic differentiation.

[arxiv paper](https://arxiv.org/pdf/2103.12604.pdf)

---------------------------------------
Diagram of the methodology proposed here.

![Diagram](Figures/Rho_ss_diagram_flat.png)

For a V-system, 

![Diagram](Figures/V_system_Kenny.png)

We maximize the transport of energy to the reaction center (RC) using gradient descent,

<!---![Diagram](Figures/Adam_opt_n_loc_gamma_d_Gamma_arrow.png)-->

![Kenny_system_GA_loc_eff](https://user-images.githubusercontent.com/32406514/139455706-96700922-f394-4ee1-b45f-0c4222abb420.gif)

All simulations were carried in [JAX](jax.readthedocs.io/). We implemented the vector-Jacobian product to differentiate the steady state to incorporate it in the framework of automatic differentiation. 
For more details about our work we created a [Google Collab](energy_transfer.ipynb), where we better exaplain the details of our work. 


[![Foo](Figures/colab_favicon_256px.png)](energy_transfer.ipynb)
---------------------------------------
**other systems**: 
We also optimized a three level quantum heat transfer model using gradient based methods. For more details, check our [arxiv paper](https://arxiv.org/pdf/2103.12604.pdf). 

![Diagram](Figures/V_system_Dvira.png)

---------------------------------------
**Prerequisites**:
[JAX](jax.readthedocs.io/)
