# Sources and third-party software

The small read-only decoder independently expresses the RAM mappings documented
by OCAtari's Asteroids extractor, not the full OCAtari wrapper. Upstream:
https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/asteroids.py
OCAtari is MIT-licensed; the license is preserved below for the
coordinate mapping reference. Emulator and ROM licensing are separate.

ALE game settings/reward definitions (GPL-2.0):
https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/games/supported/Asteroids.cpp
Native environment documentation:
https://ale.farama.org/environments/asteroids/

No arbitrary ROM packs, downloaded executables, or external study-directory
imports are required at runtime. The installed ale-py asset is used under its
own terms; this adaptor does not imply a separate ROM redistribution license.

## OCAtari MIT notice (release 2.2.1)

MIT License

Copyright (c) 2023 Quentin Delfosse, Jannis Blüml, Belal Alkufri, Bjarne Gregori, Christopher Schubert, Sebastian Gräfe, Timo Holst

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
