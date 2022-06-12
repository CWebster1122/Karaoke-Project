# Karaoke-Project
This project will make karaoke tracks out of music files.

The general scope for this project is:
1) Import songs from a Postgresql music database (MusicBrainz).
2) Cut the songs up into smaller time segments (if necessary).
3) Use FFT to produce frequency-space data.
4) Combine frequency-space and time-space waveform data into spectrograms.
5) Since a spectrogram is basically an image, send spectrograms through an image recognition CNN to separate instrumentals from voices. Need to save instrumental and voice data (keeping track of time).
6) Send voice data through a speech recognition neural network, and then create slides of text (karaoke slides) with an upper limit on how much text on each slide. 
7) Combine the speech recognition data with time data in order to highlight lyrics on the karaoke slides in real-time.
 
