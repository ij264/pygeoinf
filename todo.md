# Tasks that require more brain power.
- [ ] Validate the log-evidence logic.
## Log-Evidence
It looks like the evidence just keeps rising. There are two possible explanations for this:
1. There are too many data points and we have not hit the limits of what the data can resolve.

TEST: Vary the number of data points and see if a turning point arises.

2. The prior scale may be too small, which leads higher degrees being required to achieve sufficient amplitude to fit the data.

We can check the second issue by comparing the standard deviation of our data $d$ with the standard deviation of the prior.

TEST:

-- [x] Compare relative sizes of data and prior standard deviation. If data_std is much larger than the prior scale, the model cannot "reach" the data.

The prior std and data std are quite comparable, which implies that the data is the issue. However, we only have 50 data points at the moment.

There is a slight peak around L_max = 15, but the RMSE drops below the STD at around 15, which is more conclusive.

 -- [ ] In order to have a sharper peak, try increasing the prior order from 2. to 4.

- [x] Construct the preconditioner
-- At the moment, I am constructing the preconditioner by doing an lmax of 16. I am unable to construct the preconditioner by parallelising though, for now.
But wow! It is so much faster. With preconditioner: ~ 5 s. Without ~ 1 minute.
The prior order (for a fixed scale) controls how the power distribution of the signal decays off as a function of degree.

The model space, as long as it is appropriate, doesn't make a huge difference.
Two model spaces, given the same prior, converge to the same result.
# PyGeoInf
- [] Add functionality to calculate the evidence.
- [] Add functionality to calculate the power for a model.
# Real data
- [ ] Try calculating the evidence for the real data.

# Meeting with David
- [ ] Move the power to the SphereHelper class.
-- [ ] If I want just a specific degree, the push forward operator is the way forward.
-- [ ] Implement `chunking'.
- [ ] Add a method that plots a 2D historgram, which is in the SphereHelper class.
- [ ]
# Smaller tasks that can be done in < 5 min breaks.
- [ ] Restructure the pygeoinf folder.

- [ ] Create a 'Things to do within a couple days of baby arriving.' Add 'Apply for nursery place at the University.'


[ ] Pick a prior amplitude of 10 km, and a prior length scale of 200 km. Look at the power spectrum and see how sensitive the the power spectrum is. Picking a high amplitude and rough scale,
- [x] Focus on the lower degrees. What we think is that the higher degrees is that the prior dominates.
- [x] Generate the degree specific histogram.
- [x] Regularly pull from main to see if there are any conflicts.
- [] Fix log evidence.
Dynamic topography is driven by convection, and in convecting systems, the density anomalies organise themselves according to the governing equations. Matt Lees. Take a look at their paper to see the spectra.