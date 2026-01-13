# Ekert-91 variant simulations

Language: python

External dependencies: numpy, random, matplotlib

There is also a 'library'(ahem) `e91lib.py`, which contains some helper functions I wrote for the simulations. 

## Content

`simulations` contains the bulk of the simulations involving alternate measurement bases(3, 4, 5, 6 have been implemented thus far), as well as basically everything else. It is also the most well-maintained and documented, as our research scope pivoted from statistical deviation analyses(which was the usage purpose of the other two folders) to expanded measurement bases.

`models`(DEPRECATED) contain base simulations I wrote at the beginning. It contains variants of the base case E91 with 3 measurement angles.

`model-devs` (DEPRECATED) contains simulations and some data for calculating & plotting the deviations of error margins, as well as the key rates, with or without noise.

