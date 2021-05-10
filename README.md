# Advanced cognitive modeling

<img src="https://github.com/LegrandNico/CognitiveModeling/raw/master/images/network.png" align="right" alt="metadPy" height="250" HSPACE=30>

This repository contains material for the advanced cognitive modeling course (Aarhus University). All Monday will be allocated to lectures, the practice and applications will be on Fridays. We will use Python, [PyMC3](https://docs.pymc.io/) for Bayesian modelling, [Tensorflow](https://www.tensorflow.org/) and [OpenGym](https://gym.openai.com/) for deep/reinforcement learning).

**Prerequisites:** This course will be run using Python. Being familiar with variables, lists, dicts, the numpy and scipy libraries as well as plotting in matplotlib is required. If you have never programmed in Python, or if you have limited experience, you might consider preparing with the following tutorials:
* [Software carpentry 1-day Python tutorial](https://swcarpentry.github.io/python-novice-inflammation/)
* [Scipy Lecture Notes](https://scipy-lectures.org/)
* [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

The portfolio will consist of 3 Jupyter notebook exercises (Bayesian modeling on weeks 11, Reinforcement learning on week 16 and Deep reinforcement learning on week 19).

# Slides
* [Introduction](https://github.com/LegrandNico/CognitiveModeling/raw/master/Slides/Advanced%20cognitive%20modeling%20%E2%80%93%20%201.1%20Introduction.pdf)
* [Bayesian Modeling](https://github.com/LegrandNico/CognitiveModeling/raw/master/Slides/Advanced%20cognitive%20modeling%20%E2%80%93%20%202.Bayesian%20modeling.pdf)
* [Reinforcement Learning](https://github.com/LegrandNico/CognitiveModeling/raw/master/Slides/Advanced%20cognitive%20modeling%20%E2%80%93%20%203.Reinforcement%20learning.pdf)

# Notebooks

## Bayesian modeling

| Notebook | Colab | nbViewer |
| --- | ---| --- |
| Coin-flipping problem - Bayes' rule | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/0-BayesRule.ipynb) |  [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/raw/master/notebooks/0-BayesRule.ipynb)
| Thinking probabilistically | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/1-ThinkingProbabilistically.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/raw/master/notebooks/1-ThinkingProbabilistically.ipynb) 
| Introduction to PyMC3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/2-IntroductionPyMC3.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/raw/master/notebooks/2-IntroductionPyMC3.ipynb)
| Normal distribution- Linear Regression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/2-LinearRegression.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/raw/master/notebooks/2-LinearRegression.ipynb)
| 7 scientists problem - Measurement of IQ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/3-sevenScientistsIQ.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/raw/master/notebooks/3-sevenScientistsIQ.ipynb)
| Psychophysics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/4-Psychophysics.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/blob/master/notebooks/4-Psychophysics.ipynb)
| Exam scores | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/5-ExamScores.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/blob/master/notebooks/5-ExamScores.ipynb)
| Memory retention | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/6-MemoryRetention.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/blob/master/notebooks/6-MemoryRetention.ipynb)
| Model Comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/7-ModelComparison.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/blob/master/notebooks/7-ModelComparison.ipynb)
| Comparing gaussian means | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/8-ComparingGaussianMeans.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/blob/master/notebooks/8-ComparingGaussianMeans.ipynb)
| GLM | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/9-GLM.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/blob/master/notebooks/9-GLM.ipynb)
| Mixture models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/10-MixtureModels.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/blob/master/notebooks/10-MixtureModels.ipynb)

## Reinforcement learning

| Notebook | Colab | nbViewer |
| --- | ---| --- |
| OpenAI Gym | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/11-OpenAIGym.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/blob/master/notebooks/11-OpenAIGym.ipynb)
| Q-learning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/12-Q-learning.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/blob/master/notebooks/12-Q-learning.ipynb)
| Deep Q-learning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/13-DeepQ-learning.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/blob/master/notebooks/13-DeepQ-learning.ipynb)

# Portfolios

| Notebook | Colab | nbViewer |
| --- | ---| --- |
| Portfolio 1 - deadline:  03.22.2021 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/portfolio1.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/blob/master/notebooks/portfolio1.ipynb)
| Portfolio 2 - deadline:  04.30.2021 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/portfolio2.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/blob/master/notebooks/portfolio2.ipynb)
| Portfolio 3 - | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/portfolio3.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/blob/master/notebooks/portfolio3.ipynb)
# Schedule

---
## Introduction
---

### Week 5

| Friday | 12:00 – 14:00 | Presentation – Introduction to cognitive and computational modelling | 
| --- | ---| --- |

* Neuromatch Academy (W1D1 - Intro) - [Video](https://www.youtube.com/watch?v=KxldhMR5PxA) - [Slides](https://osf.io/rbx2a/?direct%26mode=render%26action=download%26mode=render)
* Neuromatch Academy (W1D1 - Outro) - [Video](https://www.youtube.com/watch?v=KZQXfQL1SH4) - [Slides](https://osf.io/9hkg2/?direct%26mode=render%26action=download%26mode=render)
* Neuromatch Academy (W1D1 - Tutorials) - [Videos](https://www.youtube.com/watch?v=KgqR_jbjMQg&list=PLkBQOLLbi18ObAiSOZ42YBwOQIKNvspeI) - [Slides](https://osf.io/6dxwe/?direct%26mode=render%26action=download%26mode=render)

* Lei Zhang - BayesCog Summer 2020 Lecture 09 - [Intro to cognitive modeling & Rescorla-Wagner model](https://youtu.be/tXFKYWx6c3k?list=PLfRTb2z8k2x9gNBypgMIj3oNLF8lqM44-)

>Huys, Q. J. M., Maia, T. V., & Frank, M. J. (2016). Computational psychiatry as a bridge from neuroscience to clinical applications. *Nature Neuroscience, 19(3), 404–413*. https://doi.org/10.1038/nn.4238

>Kriegeskorte, N., & Douglas, P. K. (2018). Cognitive computational neuroscience. *Nature Neuroscience, 21(9), 1148–1160*. https://doi.org/10.1038/s41593-018-0210-5 

>Lewandowsky, S. & Farrell, S. (2011). Computational modeling in cognition : principles and practice. Thousand Oaks: Sage Publications. *Chapter 1*.

>Forstmann, B. & Wagenmakers. (2015). An introduction to model-based cognitive neuroscience. New York, NY: Springer. *Chapter 1: An Introduction to Cognitive Modeling*.

---

### Week 6

| Monday | 14:00 – 16:00 | The process of modeling
| --- | ---| --- |
| Friday | 12:00 – 14:00 | Programming Probabilistically – Bayes' rule, distributions 

>Bodner, K., Brimacombe, C., Chenery, E. S., Greiner, A., McLeod, A. M., Penk, S. R., & Vargas Soto, J. S. (2021). Ten simple rules for tackling your first mathematical models: A guide for graduate students by graduate students. PLOS Computational Biology, 17(1), e1008539. https://doi.org/10.1371/journal.pcbi.1008539

> >Wilson, R. C., & Collins, A. G. (2019). Ten simple rules for the computational modeling of behavioral data. ELife, 8. https://doi.org/10.7554/elife.49547

>Blohm, G., Kording, K. P., & Schrater, P. R. (2020). A How-to-Model Guide for Neuroscience. Eneuro, 7(1), ENEURO.0352-19.2019. https://doi.org/10.1523/eneuro.0352-19.2019

* Neuromatch Academy (W1D2 - Intro) - [Video](https://www.youtube.com/watch?v=8pz_NH5_Zy4) - [Slides](https://osf.io/kmwus/?direct%26mode=render%26action=download%26mode=render)
* Neuromatch Academy (W1D2 - Outro) - [Video](https://www.youtube.com/watch?v=Il8zOmCMFAA) - [Slides](https://osf.io/agrp6/?direct%26mode=render%26action=download%26mode=render)
* Neuromatch Academy (W1D2 - Tutorials) - [Videos](https://youtu.be/x4b2-hZoyiY?list=PLkBQOLLbi18Nc7rjBNO99bZQyuTY0TAcE) - [Slides](https://osf.io/kygfn/?direct%26mode=render%26action=download%26mode=render)

---
## Bayesian modeling
---
### Week 7

| Monday | 14:00 – 16:00 | Introduction to PyMC3 – MCMC – Parameter estimation |
| --- | ---| --- |
| Friday | 12:00 – 14:00 | Normal distributions - Linear regression |

This week we will focus on Chapters 1 and 2 from the book (Bayesian analysis with Python).

> Martin, O. (2018). Bayesian analysis with Python : introduction to statistical modeling and probabilistic programming using PyMC3 and ArviZ. Birmingham, UK: Packt Publishing. *Chapter 1 and 2*.

#### Additional references and videos

> Kruschke, J. (2015). Doing Bayesian data analysis : a tutorial with R, JAGS, and Stan. Boston: Academic Press. *Chapter 6 and 7*.

> Pilon, C. Bayesian methods for hackers : probabilistic programming and Bayesian inference. New York: Addison-Wesley. *Chapter 1 and 2*.

 * An introduction to Markov Chain Monte Carlo using PyMC3 by Chris Fonnesbeck [first half of the conference] - [Video](https://www.youtube.com/watch?v=SS_pqgFziAg) - [Code](https://github.com/fonnesbeck/mcmc_pydata_london_2019/tree/master/notebooks)


---

### Week 8
| Monday | 14:00 – 16:00 | – 7 scientists - Measurement of IQ |
| --- | ---| --- |
| Friday | 12:00 – 14:00 | Psychophysics |

> Martin, O. (2018). Bayesian analysis with Python : introduction to statistical modeling and probabilistic programming using PyMC3 and ArviZ. Birmingham, UK: Packt Publishing. *Chapter 3*.

#### Additional references and videos

> Kruschke, J. (2015). Doing Bayesian data analysis : a tutorial with R, JAGS, and Stan. Boston: Academic Press. *Chapter 9*.
> 
* BayesCog Summer 2020 Lecture 11 - Hierarchical Bayesian modeling - [Video](https://www.youtube.com/watch?v=pCIsGBbUCCE&list=PLfRTb2z8k2x9gNBypgMIj3oNLF8lqM44-&index=11)

---

### Week 9
| Monday | 14:00 – 16:00 | Hierarchical Bayesian modelling |
| --- | ---| --- |
| Friday | 12:00 – 14:00 | Memory retention |

> Kruschke, J. (2015). Doing Bayesian data analysis : a tutorial with R, JAGS, and Stan. Boston: Academic Press. *Chapter 10*.

* Bayesian modeling without the math: An introduction to PyMC3- [Video](https://www.youtube.com/watch?v=uxGhjXS3ILE&feature=youtu.be)

---

### Week 10

|  |  | Course content |
| --- | ---| --- |
| Monday | 14:00 – 16:00 | **Models comparison** Ch.5 (part I) - Comparing linear models |
| Friday | 12:00 – 14:00 | **Models comparison** Ch.5 (part II) - Bayes factors, exercises with group difference, one sample t test, repeated measures |

> Martin, O. (2018). Bayesian analysis with Python : introduction to statistical modeling and probabilistic programming using PyMC3 and ArviZ. Birmingham, UK: Packt Publishing. *Chapter 5*.

>Lee, M. & Wagenmakers. (2013). Bayesian cognitive modeling : a practical course. Cambridge New York: Cambridge University Press. *Chapter 8*.
#### Additional references and videos

> van de Schoot, R., Depaoli, S., King, R., Kramer, B., Märtens, K., Tadesse, M. G., Vannucci, M., Gelman, A., Veen, D., Willemsen, J., & Yau, C. (2021). Bayesian statistics and modelling. Nature Reviews Methods Primers, 1(1). https://doi.org/10.1038/s43586-020-00001-2

* The Bayesian Workflow: Building a COVID-19 Model by Thomas Wiecki [Part 1] - [Video](https://www.youtube.com/watch?v=ZxR3mw-Znzc)
* BayesCog Summer 2020 Lecture 12 - Model comparison - [Video](https://www.youtube.com/watch?v=xmt_H2q2tO8&list=PLfRTb2z8k2x9gNBypgMIj3oNLF8lqM44-&index=12)
* Intro to Bayesian Model Evaluation, Visualization, & Comparison Using ArviZ | SciPy 2019 Tutorial - [Video](https://www.youtube.com/watch?v=bmWMdVQlzIA)

---

### Week 11
|  |  | Course content |
| --- | ---| --- |
| Monday | 14:00 – 16:00 | **Generalizing linear models** - Course and live coding covering Ch.4 and exercises on using GLM during the second hour |
| Friday | 12:00 – 14:00 | **Mixture models** - Course and live coding covering Ch.6 and exercises with mixture models during the second hour |

> Martin, O. (2018). Bayesian analysis with Python : introduction to statistical modeling and probabilistic programming using PyMC3 and ArviZ. Birmingham, UK: Packt Publishing. *Chapter 4 & 6*.

#### Additional references and videos
> > Kruschke, J. (2015). Doing Bayesian data analysis : a tutorial with R, JAGS, and Stan. Boston: Academic Press. *Chapter 15*.

[GLM - Linear regression](https://docs.pymc.io/notebooks/GLM-linear.html) (PyMC3 documentation)
[GLM - Robust linear regression](https://docs.pymc.io/notebooks/GLM-robust.html) (PyMC3 documentation)
[GLM - Hierarchical linear regression](https://docs.pymc.io/notebooks/GLM-hierarchical.html) (PyMC3 documentation)

---

### Week 12
|  |  | Course content |
| --- | ---| --- |
| Monday | 14:00 – 16:00 | **Time series** - Introduction to Markov processes and hidden markov models for sequential data analysis in base Python |
| Friday | 12:00 – 14:00 | **Time series** - Hidden Markov models using PyMC3 |

* Markov Models From The Bottom Up, with Python, Eric Ma - [link](https://ericmjl.github.io/essays-on-data-science/machine-learning/markov-models/)

#### Additional references and videos

* Hierarchical Time Series With Prophet and PyMC3 by Matthijs Brouns - [Video](https://www.youtube.com/watch?v=appLxcMLT9Y)

* Scott Linderman, Machine Learning Methods for Neural Data Analysis, 2021 Stanford University - [Github repos](https://github.com/slinderman/stats320)

* Neuromatch, W2D3 Decision Making Intro - [Video](https://www.youtube.com/watch?v=bJIAWgycuVU)

* Ankan, A. & Panda, A. (2018). Hands-On Markov Models with Python. Birmingham: Packt Publishing.
  
* https://github.com/LegrandNico/hmm-mne
---

### Week 13 - No class

---

## Reinforcement learning
For the (deep) reinforcement learning part of the course, we will be using *Reinforcement leasrning: An introduction* *(Sutton & Barto, 2018)*. You can download the book [here](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf). The core concepts of reinforcement learning are nicely introduced in Chris Willcoks' Reinforcement learning course (see the [Videos](https://www.youtube.com/playlist?list=PLMsTLcO6ettgmyLVrcPvFLYi2Rs-R4JOE) and [Slides](https://cwkx.github.io/teaching.html). The Neuromatch academy session focused on reinforcement learning is also highly recommended (see course material [here](https://www.neuromatchacademy.org/syllabus)). You can find Colab notebooks accompagning all these courses online.

---
### Week 14
|  |  | Course content |
| --- | ---| --- |
| Friday | 12:00 – 14:00 | **Introduction to reinforcement learning** - Key concepts of reiinforcement learning and deep reinforcement learning. Introduction to the OpenAI Gym environment. Going through the first part of the chapter during the second hour.

#### Additional references and videos

* Neuromatch Reinforcement Learning Intro - Doina Precup [Video](https://www.youtube.com/watch?v=abEarxx6Kgc)
* Neuromatch Reinforcement Learning Outro - Tim Behrens [Video](https://www.youtube.com/watch?v=abEarxx6Kgc)
* Reinforcement Learning: Machine Learning Meets Control Theory - [Video](https://www.youtube.com/watch?v=0MNVhXEX9to)
* Deep Reinforcement Learning: Neural Networks for Learning Control Laws - [Video](https://www.youtube.com/watch?v=IUiKAD6cuTA)
* Reinforcement Learning 1: Foundations - Chris Willcocks - [Video](https://www.youtube.com/watch?v=K67RJH3V7Yw&list=PLMsTLcO6ettgmyLVrcPvFLYi2Rs-R4JOE&index=2) - [Slides](https://cwkx.github.io/data/teaching/dl-and-rl/rl-lecture1.pdf)

---

### Week 15
|  |  | Course content |
| --- | ---| --- |
| Monday | 14:00 – 16:00 | OpeAI Gym |
| Friday | 12:00 – 14:00 | **Markov Decision Process**. Bellman Equations. |

> Géron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow : concepts, tools, and techniques to build intelligent systems. Sebastopol, CA: O'Reilly Media, Inc. *Ch.18. Reinforcement learning*.

> Sutton, R. & Barto, A. (2018). Reinforcement learning : an introduction. Cambridge, Massachusetts London, England: The MIT Press. *Ch.3. Finite Markov Decision Processes*

#### Additional references and videos
* Reinforcement Learning 2: Markov Decision Processes - Chris Willcocks - [Video](https://www.youtube.com/watch?v=RmOdTQYQqmQ&list=PLMsTLcO6ettgmyLVrcPvFLYi2Rs-R4JOE&index=2)
* Reinforcement Learning 3: OpenAI Gym - Adam Leach - [Video](https://www.youtube.com/watch?v=BNSwFURmaCA&list=PLMsTLcO6ettgmyLVrcPvFLYi2Rs-R4JOE&index=3)

>Pulcu, E., & Browning, M. (2019). The Misestimation of Uncertainty in Affective Disorders. *Trends in Cognitive Sciences, 23(10), 865–875*. https://doi.org/10.1016/j.tics.2019.07.007 

>Botvinick, M., Ritter, S., Wang, J. X., Kurth-Nelson, Z., Blundell, C., & Hassabis, D. (2019). Reinforcement Learning, Fast and Slow. *Trends in Cognitive Sciences, 23(5), 408–422*. https://doi.org/10.1016/j.tics.2019.02.006

---

### Week 16
|  |  | Course content |
| --- | ---| --- |
|Monday | 14:00 – 16:00 | **Markov Decision Process** Dynamic programming. Value and Policy iteration algorithms. |            
|Friday | 12:00 – 14:00 | **Markov Decision Process** Frozen lake environment. Applications to computational psychiatry Zorowitz et al. (2020)

> Sutton, R. & Barto, A. (2018). Reinforcement learning : an introduction. Cambridge, Massachusetts London, England: The MIT Press. *Ch.4. Dynamic Programming*

#### Additional references and videos

* Reinforcement Learning 4: Dynamic Programming - Chris Willcocks - [Video](https://www.youtube.com/watch?v=gqC_p2XWpLU) - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/cwkx/670c8d44a9a342355a4a883c498dbc9d/dynamic-programming.ipynb)

> Juechems, K., & Summerfield, C. (2019). Where Does Value Come From? Trends in Cognitive Sciences, 23(10), 836–850. https://doi.org/10.1016/j.tics.2019.07.012

> Zorowitz, S., Momennejad, I., & Daw, N. D. (2020). Anxiety, Avoidance, and Sequential Evaluation. *Computational Psychiatry, 4(0), 1*. https://doi.org/10.1162/cpsy_a_00026

---

### Week 17
|  |  | Course content |
| --- | ---| --- |
| Monday | 14:00 – 16:00 | Q-learning |

---

### Week 18
|  |  | Course content |
| --- | ---| --- |
| Monday | 14:00 – 16:00 | Deep Q-learning - CartPole example
| Friday | 12:00 – 14:00 | Deep Q-learning - Portfolio 2 (correction)

> Géron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow : concepts, tools, and techniques to build intelligent systems. Sebastopol, CA: O'Reilly Media, Inc. *Ch.18. Reinforcement learning*.

MIT 6.S091: Introduction to Deep Reinforcement Learning (Deep RL) - [Video](https://www.youtube.com/watch?v=zR11FLZ-O9M)

---

### Week 19
|  |  | Course content |
| --- | ---| --- |
| Monday | 14:00 – 16:00 | Deep Q-learning
| Wednesday | 12:00 – 14:00 | **Invited lecture**: Joshua Skeve - National Inequality and Individual Readiness to Cooperate

> Skewes, J. (2020, October 29). National Inequality and Individual Readiness to Cooperate. https://doi.org/10.31234/osf.io/f79rw
>  Fischbacher, Urs, and Simon Gächter. 2010. "Social Preferences, Beliefs, and the Dynamics of Free Riding in Public Goods Experiments." American Economic Review, 100 (1): 541-56. 