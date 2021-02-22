# Advanced cognitive modeling

<img src="https://github.com/LegrandNico/CognitiveModeling/raw/master/images/network.png" align="right" alt="metadPy" height="250" HSPACE=30>

This repository contains material for the advanced cognitive modeling course (Aarhus University). All Monday will be allocated to lectures, the practice and applications will be on Fridays. We will use Python, [PyMC3](https://docs.pymc.io/) for Bayesian modelling, [Tensorflow](https://www.tensorflow.org/) and [OpenGym](https://gym.openai.com/) for deep/reinforcement learning).

**Prerequisites:** This course will be run using Python. Being familiar with variables, lists, dicts, the numpy and scipy libraries as well as plotting in matplotlib is required. If you have never programmed in Python, or if you have limited experience, you might consider preparing with the following tutorials:
* [Software carpentry 1-day Python tutorial](https://swcarpentry.github.io/python-novice-inflammation/)
* [Scipy Lecture Notes](https://scipy-lectures.org/)
* [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

The portfolio will consist of 3 Jupyter notebook exercises (Bayesian modeling on weeks 11, Reinforcement learning on week 16 and Deep reinforcement learning on week 19).

## Schedule

### Week 5

| Friday | 12:00 – 14:00 | Presentation – Introduction to cognitive and computational modelling | [Slides](https://github.com/LegrandNico/CognitiveModeling/raw/master/Slides/Advanced%20cognitive%20modeling%20%E2%80%93%20%201.1%20Introduction.pdf)
| --- | ---| --- | --- |

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

**Exercises:** [Notebook](https://github.com/LegrandNico/CognitiveModeling/blob/master/notebooks/1-ThinkingProbabilistically.ipynb) - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/1-ThinkingProbabilistically.ipynb) - [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/raw/master/notebooks/1-ThinkingProbabilistically.ipynb)

>Bodner, K., Brimacombe, C., Chenery, E. S., Greiner, A., McLeod, A. M., Penk, S. R., & Vargas Soto, J. S. (2021). Ten simple rules for tackling your first mathematical models: A guide for graduate students by graduate students. PLOS Computational Biology, 17(1), e1008539. https://doi.org/10.1371/journal.pcbi.1008539

>Blohm, G., Kording, K. P., & Schrater, P. R. (2020). A How-to-Model Guide for Neuroscience. Eneuro, 7(1), ENEURO.0352-19.2019. https://doi.org/10.1523/eneuro.0352-19.2019

* Neuromatch Academy (W1D2 - Intro) - [Video](https://www.youtube.com/watch?v=8pz_NH5_Zy4) - [Slides](https://osf.io/kmwus/?direct%26mode=render%26action=download%26mode=render)
* Neuromatch Academy (W1D2 - Outro) - [Video](https://www.youtube.com/watch?v=Il8zOmCMFAA) - [Slides](https://osf.io/agrp6/?direct%26mode=render%26action=download%26mode=render)
* Neuromatch Academy (W1D2 - Tutorials) - [Videos](https://youtu.be/x4b2-hZoyiY?list=PLkBQOLLbi18Nc7rjBNO99bZQyuTY0TAcE) - [Slides](https://osf.io/kygfn/?direct%26mode=render%26action=download%26mode=render)

---

### Week 7

| Monday | 14:00 – 16:00 | Introduction to PyMC3 – MCMC – Parameter estimation |
| --- | ---| --- |
| Friday | 12:00 – 14:00 | Normal distributions - Linear regression |

This week we will focus on Chapters 1 and 2 from the book (Bayesian analysis with Python).

**Introduction to PyMC3:** [Notebook](https://github.com/LegrandNico/CognitiveModeling/blob/master/notebooks/IntroductionPyMC3.ipynb) - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/IntroductionPyMC3.ipynb) - [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/raw/master/notebooks/IntroductionPyMC3.ipynb)

**Exercises + Solutions - 1:** [Notebook](https://github.com/LegrandNico/CognitiveModeling/blob/master/notebooks/1-Solutions.ipynb) - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/1-Solutions.ipynb) - [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/raw/master/notebooks/1-Solutions.ipynb)

**Exercises - 2:** [Notebook](https://github.com/LegrandNico/CognitiveModeling/blob/master/notebooks/2-Exercises.ipynb) - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/2-Exercises.ipynb) - [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/raw/master/notebooks/2-Exercises.ipynb)

> Martin, O. (2018). Bayesian analysis with Python : introduction to statistical modeling and probabilistic programming using PyMC3 and ArviZ. Birmingham, UK: Packt Publishing. *Chapter 1 and 2*.

#### Additional references and videos

> Kruschke, J. (2015). Doing Bayesian data analysis : a tutorial with R, JAGS, and Stan. Boston: Academic Press. *Chapter 6 and 7*.

> Pilon, C. Bayesian methods for hackers : probabilistic programming and Bayesian inference. New York: Addison-Wesley. *Chapter 1 and 2*.

 * An introduction to Markov Chain Monte Carlo using PyMC3 by Chris Fonnesbeck [first half of the conference] - [Video](https://www.youtube.com/watch?v=SS_pqgFziAg) - [Code](https://github.com/fonnesbeck/mcmc_pydata_london_2019/tree/master/notebooks)


---

### Week 8
| Monday | 14:00 – 16:00 | – 7 scientist - Measurement of IQ |
| --- | ---| --- |
| Friday | 12:00 – 14:00 | Hierarchical Bayesian modeling |

> Martin, O. (2018). Bayesian analysis with Python : introduction to statistical modeling and probabilistic programming using PyMC3 and ArviZ. Birmingham, UK: Packt Publishing. *Chapter 3*.


**Exercises - 3:** [Notebook](https://github.com/LegrandNico/CognitiveModeling/blob/master/notebooks/3-Exercises.ipynb) - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/CognitiveModeling/blob/master/notebooks/3-Exercises.ipynb) - [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/CognitiveModeling/raw/master/notebooks/3-Exercises.ipynb)

#### Additional references and videos

> Kruschke, J. (2015). Doing Bayesian data analysis : a tutorial with R, JAGS, and Stan. Boston: Academic Press. *Chapter 9*.
> 
* BayesCog Summer 2020 Lecture 11 - Hierarchical Bayesian modeling - [Video](https://www.youtube.com/watch?v=pCIsGBbUCCE&list=PLfRTb2z8k2x9gNBypgMIj3oNLF8lqM44-&index=11)

---

### Week 9
| Monday | 14:00 – 16:00 | Hierarchical Bayesian modelling - II |
| --- | ---| --- |
| Friday | 12:00 – 14:00 | 8 schools problem |

> Kruschke, J. (2015). Doing Bayesian data analysis : a tutorial with R, JAGS, and Stan. Boston: Academic Press. *Chapter 10*.
> 
---

### Week 10
| Monday | 14:00 – 16:00 | Models comparison |
| --- | ---| --- |
| Friday | 12:00 – 14:00 | Models comparison |

> van de Schoot, R., Depaoli, S., King, R., Kramer, B., Märtens, K., Tadesse, M. G., Vannucci, M., Gelman, A., Veen, D., Willemsen, J., & Yau, C. (2021). Bayesian statistics and modelling. Nature Reviews Methods Primers, 1(1). https://doi.org/10.1038/s43586-020-00001-2

> Martin, O. (2018). Bayesian analysis with Python : introduction to statistical modeling and probabilistic programming using PyMC3 and ArviZ. Birmingham, UK: Packt Publishing. *Chapter 5*.

* The Bayesian Workflow: Building a COVID-19 Model by Thomas Wiecki [Part 1] - [Video](https://www.youtube.com/watch?v=ZxR3mw-Znzc)
* BayesCog Summer 2020 Lecture 12 - Model comparison - [Video](https://www.youtube.com/watch?v=xmt_H2q2tO8&list=PLfRTb2z8k2x9gNBypgMIj3oNLF8lqM44-&index=12)


---

### Week 11
| Monday | 14:00 – 16:00 | Introduction to reinforcement learning |
| --- | ---| --- |
| Friday | 12:00 – 14:00 | Rescorla-Wagner model |

* Reinforcement Learning: Machine Learning Meets Control Theory - [Video](https://www.youtube.com/watch?v=0MNVhXEX9to)

---

>Wilson, R. C., & Collins, A. G. (2019). Ten simple rules for the computational modeling of behavioral data. ELife, 8. https://doi.org/10.7554/elife.49547

### Week 12
| Monday | 14:00 – 16:00 | Reinforcement learning |
| --- | ---| --- |
| Friday | 12:00 – 14:00 | Introduction to Tensorflow - Multiarmed bandit |


---

### Week 13 - No class

---

### Week 14
| Friday | 12:00 – 14:00 | Introduction to Open AI Gym + Tensorflow
| --- | ---| --- |

---

### Week 15
| Monday | 14:00 – 16:00 | Reinforcement learning applications |
| --- | ---| --- |
| Friday | 12:00 – 14:00 | Case study: Wise and Dolan (2020) |

---

### Week 16
|Monday | 14:00 – 6:00 | Deep reinforcement learning               
| --- | ---| --- |
|Friday | 12:00 – 14:00 | Deep reinforcement learning with Tensorflow               

---

### Week 17
| Monday | 14:00 – 16:00 | Deep reinforcement learning applications |
| --- | ---| --- |

---

### Week 18
| Monday | 14:00 – 16:00 | Recurrent neural networks (RNN)               
| --- | ---| --- |
| Friday | 12:00 – 14:00 | Introduction to PsychRNN               

---

### Week 19
| Monday | 14:00 – 16:00 | Recurrent neural networks application
| --- | ---| --- |
| Wednesday | 12:00 – 14:00 | PsychRNN
 