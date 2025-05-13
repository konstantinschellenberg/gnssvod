# Scrapbook for task withing `gnssvod`

## Notes

- RINEX data worthless, no CN0 data included
- Ground data (doy 149, 2022) onwards corrupt? no obs file expelled, only nav
- What happened DOY 150 (2022)?
  - binex file size increases
  - using `convbin`, only nav file is created, no obs. Even on explicit request
  - RINEX files from Sami don't contain SNR in both, before and after DOY 150 

`urllib.error.URLError: <urlopen error [Errno 104] Connection reset by peer>`
## Todos

- [x] Make Binex converter
- [x] Process towards RINEX
- [x] installed gfzrnx: Does not support BINEX conversion
- [x] installing teqc
- [ ] Implement teqc pipeline
- [ ] Check missing ground obs data
- [ ] Multiprocessing not yet working: Results differ when redone...

## Comm
- Ask Benjamin L. if they were successful with RINEX proc

## Features
- BINEX –> RINEX converter

## Binex DRIVERS

### RTKLIB

Use RTKLIB to convert BINEX to RINEX

```bash
Installation:
wget https://github.com/tomojitakasu/RTKLIB/archive/refs/tags/2.4.3.b34L-pre0.tar.gz
tar -xzf 2.4.3.b34L-pre0.tar.gz

cd app/convbin
make
sudo make install

convbin is now an executable.
```

### teqc

Download from [UNAVCO](https://www.unavco.org/software/data-processing/teqc/teqc.html)

```bash
cp teqc /usr/local/bin
chmod a+x /usr/local/bin/teqc
teqc -version
```

