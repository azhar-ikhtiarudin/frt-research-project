# Research Internship Project at Australian National University
- Azhar Ikhtiarudin (Institut Teknologi Bandung, Indonesia)
- Project Report: https://github.com/azhar-ikhtiarudin/frt-research-project/blob/main/Project%20Report_Optimization%20of%20CYGNUS%20Readout%20Emulation%20Program.pdf

# A simple CYGNUS readout emulation toolkit

The point of this tool is to emulate a simple (but not too simple) readout of simulated TPC events. The concept is to make a toolkit, rather than a monolithic emulator, to suit the various ways in which we may want to use the tool.
For instance, we may use this toolkit to model the drift, amplification, and readout of Geant4 primary events. Or use a subset of it to take HEED + GarField drifted events and simulate only the avalanche and readout. Or we might just want to study gas diffusion, without going through a gain stage.
In practice, that means that the code works more like a python library which will be called by user code, rather than something which is used directly.

## Detector definition

The code requires the definition of a minimal set of detector attributes, for the purposes of drifting/applying gain to the readout. Note that the default units are keV, cm, volts, and seconds.

**Gas Properties**: w-value (keV / ion pair), transverse/longditudinal diffusion coefficients (cm$^2$/s), and mobility (cm$^2$V$^{-1}$s$^{-1}$)

**Detector properties**: drift field (V/cm), readout pitch along x/y (cm), and time sampling rate (seconds).

## Event structure

The basic unit that the tool works with is an 'event'. Events go through several stages:

**PrimaryEvt**: Initial energy deposit in the TPC. This is stored as a Pandas dataframe with at least the locations of the energy deposits (x,y,z,Edep) as input from some sort of primary event generator. The dataframe may also have things like the vertex location, particle type, etc. but they're not strictly necessary. If it's not provided, the toolkit will Poisson sample the W-value of the gas at each energy deposit location to generate the number of initial charge carriers.

**DriftedEvt**: This is the status of the event, upon reaching the avalanche region (assumed to be a plane along x-y with a fixed z position). The code will apply Gaussian smearing, worked out using the transverse and longditudinal diffusion coefficients, the carrier mobility, and the drift field strength. Carriers are stored individually at this stage, in a data frame with (index in PrimaryEvt, x, y, dt). Note the use of dt rather than z.

**ReadoutGrid and ReadoutEvt**: Since avalanche gain introduces a factor of maybe 1 million, storing individual carriers at this point makes no sense. Instead, for each drifted carrier, the gain distribution is sampled (assumed to be exponentially distributed if no other info is given, as in Sauli's textbook, pg 151). Then a point spread function is scaled by that number of electrons, and added to a binned readout.
