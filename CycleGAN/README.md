# CyleGAN (Cycle-Consistent Adversarial Network)
Model generatif yang dirancang untuk mentransformasi gambar dari satu domain ke domain lain tanpa memerlukan pasangan gambar selama pelatihan.
CycleGAN merupakan perkembangan dari pix2pix yang memerlukan pasangan gambar. 
## Main Ideas
### Dua Generator dan Dua Diskriminator
- G : Mentransformasi gambar dari domain X ke domain Y. Horse -> Zebra
- F : Mentransformasi gambar dari domain Y ke domain X. Zebra -> Horse
- D_X : Diskriminator yang membedakan antara gambar asli dari domain X dan gambar hasil transformasi dari F. Mencari tahu apakah Real Horse atau Fake Horse.
- D_Y : Diskriminator yang membedakan antara gambar asli dari domain Y dan gambar hasil transformasi dari G. Mencari tahu apakah Real Zebra atau Fake Zebra.
### Cycle Consistency Loss
- Ketika kita mentransformasi gambar dari X ke Y (G(X)), lalu mentransformasinya kembali ke X (F(G(X))) sehingga menjadi $\hat{X}$, hasilnya harus mirip dengan gambar X asli.
- Begitu pula sebaliknya, transformasi gambar dari Y ke X (F(Y)), lalu ditransformasi kembali ke Y (G(F(Y))), maka $\hat{Y}$ harus mirip dengan gambar Y asli.
- Cycle Consistency Loss memastikan bahwa transformasi dua arah ini konsisten dan tidak kehilangan informasi penting.
### Loss (Full)
- $$\mathcal{L}(G, F, D_X, D_Y) = \mathcal{L}_{GAN}(G, D_Y, X, Y) + \mathcal{L}_{GAN}(F, D_X, Y, X) + \lambda \mathcal{L}_{cyc}(G, F)$$
## Notes
- Defaultnya, CycleGAN dilatih hanya dengan dua domain saja
- Jika menggunakan dataset gambar tidak berwarna dengan task mentransformasinya menjadi berwarna, dapat ditambah dengan identity loss.
- Berbeda dengan GAN vanilla, dimana untuk mengukur performa (evaluasi) menggunakan matriks 0 1 (0 false, 1 true) yang nantinya akan diukur dengan BCE Loss. CycleGAN menggunakan PatchGAN, dimana matriksnya akan responds terhadap patch-patch di gambar asli (grid).
- Alur penuh dari CycleGAN dapat dilihat [disini](./CycleGAN_flow.png). 
- Training dan scaling (loss dan gradient) dengan AMP torch.
## Experiment
### Data
Dilatih dengan data asli dari paper CycleGAN Horse-Zebra