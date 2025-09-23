# Generative Adviversal Network
Tujuan model -> Mempelajari data latih untuk menghasilkan data baru semirip mungkin dengan data asli

# Main Idea
## G - Generator
- Dimulai dari random weights
- Generator akan mempelajari data (gambar) dari data asli
- Generator mengambil noise acak atau vektor acak sebagai input dan menghasilkan sampel data

## D - Discriminator
- Discriminator akan mengambil data asli dan hasil generasi dari generator
- Discriminator akan mencoba mengklasifikasikan sampel sebagai asli atau palsu


## Training Loop
### Inisialisasi
- Generator : Mulai dengan membuat output acak (tidak bermakna)
- Discriminator : Mulai dengan kemampuan acak untuk membedakan data asli dan palsu
### Input Output
- Input (G) : Data acak dari distribusi
- Output (G) : Data palsu

- Input (D) : Campuran data asli (dataset) dan data palsu (dari Output(G))
- Output (G) : Probabilitas yang menunjukkan seberapa yakin ia bahwa inputnya adalah data asli

### Flow Sederhana
- Di setiap iterasi pelatihan, Generator akan membuat sekumpulan data palsu dari random noise
- Discriminator akan mengevaluasi keduanya (data asli dan data palsu)

## Loss
- Kedua bagian dilatih dengan loss function yang bertentangan
### Discriminator
- Memaksimalkan kemampuan membedakan data asli dan palsu
- $$Loss(D) = -[(log(D(x))) + log(1 - D(G(z)))]$$
- Intuisi : Discriminator ingin $D(x)$ mendekati 1 (benar) dan $D(G(z))$ mendekati 0 (salah)

### Generator
- Menipu discriminator agar menganggap data palsu sebagai data asli
- $$Loss(G) = -log(D(G(z)))$$
- Intuisi : Generator ingin $D(G(z))$ mendekati 1 (menipu discriminator)


## Sources
- [An Introduction to Generative Adversarial Networks (GANs)](https://medium.com/aimonks/an-introduction-to-generative-adversarial-networks-gans-454d127640c1)