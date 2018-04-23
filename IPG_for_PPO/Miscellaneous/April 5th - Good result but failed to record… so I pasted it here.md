### April 5th - Good result but failed to record… so I pasted it here

- **Compared with original PPO, this time learning is very fast, while it is not always like this.**

/Library/Frameworks/Python.framework/Versions/3.6/bin/python3 /Users/jianingsun/PycharmProjects/RL_Project/IPG_for_PPO/main.py FetchReach-v0 -n 3000
Value Params -- h1: 170, h2: 29, h3: 5, lr: 0.00186
2018-04-05 18:38:04.734432: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.2 AVX AVX2 FMA
Value Params -- h1: 170, h2: 29, h3: 5, lr: 0.00186
Policy Params -- h1: 170, h2: 82, h3: 40, lr: 9.94e-05, logvar_speed: 8
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2957: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/numpy/core/_methods.py:80: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
on_policy_loss: 4.2385526199950616e-12. Off_policy_loss: -0.22019464492797852. Total Loss: -0.22019464492373997

***** Episode 15, Mean R = -48.7 *****
BaselineLoss: 0.00287
Beta: 1.5
CriticLoss: 0.298
KL: 0.00903
PolicyEntropy: 3.68
Steps: 750
TotalLoss: -0.22


on_policy_loss: -4.2385526199950616e-12. Off_policy_loss: -0.27002510070800784. Total Loss: -0.2700251007122464

***** Episode 30, Mean R = -49.7 *****
BaselineLoss: 0.00247
Beta: 2.25
CriticLoss: 0.765
KL: 0.00865
PolicyEntropy: 3.69
Steps: 750
TotalLoss: -0.27


on_policy_loss: 5.086263262417863e-12. Off_policy_loss: -0.2221316146850586. Total Loss: -0.22213161467997233

***** Episode 45, Mean R = -49.2 *****
BaselineLoss: 0.00164
Beta: 2.25
CriticLoss: 0.55
KL: 0.00449
PolicyEntropy: 3.71
Steps: 750
TotalLoss: -0.222


on_policy_loss: -1.69542098878613e-12. Off_policy_loss: -0.19112442016601563. Total Loss: -0.19112442016771106

***** Episode 60, Mean R = -50.0 *****
BaselineLoss: 0.000861
Beta: 2.25
CriticLoss: 0.546
KL: 0.00505
PolicyEntropy: 3.71
Steps: 750
TotalLoss: -0.191


on_policy_loss: -9.324815290293978e-12. Off_policy_loss: -0.19264160156250001. Total Loss: -0.19264160157182483

***** Episode 75, Mean R = -49.3 *****
BaselineLoss: 0.000311
Beta: 2.25
CriticLoss: 0.573
KL: 0.00569
PolicyEntropy: 3.73
Steps: 750
TotalLoss: -0.193


on_policy_loss: -5.086263262417863e-12. Off_policy_loss: -0.20468645095825197. Total Loss: -0.20468645096333823

***** Episode 90, Mean R = -49.1 *****
BaselineLoss: 8.91e-05
Beta: 2.25
CriticLoss: 0.747
KL: 0.00429
PolicyEntropy: 3.73
Steps: 750
TotalLoss: -0.205


on_policy_loss: -1.2927584928471939e-11. Off_policy_loss: -0.20002321243286134. Total Loss: -0.2000232124457889

***** Episode 105, Mean R = -47.9 *****
BaselineLoss: 9.33e-05
Beta: 2.25
CriticLoss: 0.728
KL: 0.00355
PolicyEntropy: 3.72
Steps: 750
TotalLoss: -0.2


on_policy_loss: -6.357828633933119e-12. Off_policy_loss: -0.1984300422668457. Total Loss: -0.19843004227320354

***** Episode 120, Mean R = -49.5 *****
BaselineLoss: 5.11e-05
Beta: 2.25
CriticLoss: 0.725
KL: 0.00198
PolicyEntropy: 3.73
Steps: 750
TotalLoss: -0.198


on_policy_loss: 1.0596381846047128e-11. Off_policy_loss: -0.19876846313476562. Total Loss: -0.19876846312416924

***** Episode 135, Mean R = -50.0 *****
BaselineLoss: 4.26e-05
Beta: 2.25
CriticLoss: 0.735
KL: 0.00445
PolicyEntropy: 3.72
Steps: 750
TotalLoss: -0.199


on_policy_loss: -1.80668304021007e-11. Off_policy_loss: -0.1980510711669922. Total Loss: -0.19805107118505902

***** Episode 150, Mean R = -49.3 *****
BaselineLoss: 3.78e-05
Beta: 2.25
CriticLoss: 0.735
KL: 0.00586
PolicyEntropy: 3.7
Steps: 750
TotalLoss: -0.198


on_policy_loss: 3.4756131602383296e-11. Off_policy_loss: -0.19795623779296875. Total Loss: -0.1979562377582126

***** Episode 165, Mean R = -48.6 *****
BaselineLoss: 6.51e-05
Beta: 3.38
CriticLoss: 0.739
KL: 0.00679
PolicyEntropy: 3.7
Steps: 750
TotalLoss: -0.198


on_policy_loss: 1.5470716855740344e-11. Off_policy_loss: -0.19793441772460937. Total Loss: -0.19793441770913867

***** Episode 180, Mean R = -48.8 *****
BaselineLoss: 0.000154
Beta: 3.38
CriticLoss: 0.741
KL: 0.00206
PolicyEntropy: 3.7
Steps: 750
TotalLoss: -0.198


on_policy_loss: -1.2079874286049137e-11. Off_policy_loss: -0.19742317199707032. Total Loss: -0.1974231720091502

***** Episode 195, Mean R = -49.5 *****
BaselineLoss: 4.96e-05
Beta: 3.38
CriticLoss: 0.74
KL: 0.0015
PolicyEntropy: 3.71
Steps: 750
TotalLoss: -0.197


on_policy_loss: 2.1192763099975308e-12. Off_policy_loss: -0.19753406524658204. Total Loss: -0.19753406524446276

***** Episode 210, Mean R = -49.6 *****
BaselineLoss: 2.17e-05
Beta: 3.38
CriticLoss: 0.743
KL: 0.00475
PolicyEntropy: 3.72
Steps: 750
TotalLoss: -0.198


on_policy_loss: -3.39084197757226e-12. Off_policy_loss: -0.19846921920776367. Total Loss: -0.1984692192111545

***** Episode 225, Mean R = -49.5 *****
BaselineLoss: 5.41e-05
Beta: 3.38
CriticLoss: 0.749
KL: 0.00353
PolicyEntropy: 3.72
Steps: 750
TotalLoss: -0.198


on_policy_loss: -3.814697298783661e-12. Off_policy_loss: -0.1973775291442871. Total Loss: -0.1973775291481018

***** Episode 240, Mean R = -49.8 *****
BaselineLoss: 1.68e-05
Beta: 3.38
CriticLoss: 0.744
KL: 0.00181
PolicyEntropy: 3.7
Steps: 750
TotalLoss: -0.197


on_policy_loss: -1.1020235983020635e-11. Off_policy_loss: -0.19706672668457031. Total Loss: -0.19706672669559055

***** Episode 255, Mean R = -49.4 *****
BaselineLoss: 3.24e-05
Beta: 3.38
CriticLoss: 0.744
KL: 0.00222
PolicyEntropy: 3.7
Steps: 750
TotalLoss: -0.197


on_policy_loss: 1.4411078552711841e-11. Off_policy_loss: -0.19720985412597658. Total Loss: -0.1972098541115655

***** Episode 270, Mean R = -48.3 *****
BaselineLoss: 0.000111
Beta: 3.38
CriticLoss: 0.746
KL: 0.00151
PolicyEntropy: 3.67
Steps: 750
TotalLoss: -0.197


on_policy_loss: -3.814697298783661e-12. Off_policy_loss: -0.196735897064209. Total Loss: -0.1967358970680237

***** Episode 285, Mean R = -49.6 *****
BaselineLoss: 3.34e-05
Beta: 2.25
CriticLoss: 0.744
KL: 0.00137
PolicyEntropy: 3.67
Steps: 750
TotalLoss: -0.197


on_policy_loss: -8.47710494393065e-13. Off_policy_loss: -0.1967172622680664. Total Loss: -0.19671726226891412

***** Episode 300, Mean R = -49.4 *****
BaselineLoss: 1.78e-05
Beta: 2.25
CriticLoss: 0.744
KL: 0.00396
PolicyEntropy: 3.68
Steps: 750
TotalLoss: -0.197


on_policy_loss: -1.3775294978775794e-12. Off_policy_loss: -0.19636878967285157. Total Loss: -0.1963687896742291

***** Episode 315, Mean R = -48.5 *****
BaselineLoss: 5.86e-05
Beta: 2.25
CriticLoss: 0.743
KL: 0.00223
PolicyEntropy: 3.69
Steps: 750
TotalLoss: -0.196


on_policy_loss: -1.949734122301076e-11. Off_policy_loss: -0.19634429931640626. Total Loss: -0.1963442993359036

***** Episode 330, Mean R = -49.0 *****
BaselineLoss: 5e-05
Beta: 2.25
CriticLoss: 0.743
KL: 0.00537
PolicyEntropy: 3.67
Steps: 750
TotalLoss: -0.196


on_policy_loss: 8.13272234741665e-12. Off_policy_loss: -0.19115951538085937. Total Loss: -0.19115951537272666

***** Episode 345, Mean R = -48.6 *****
BaselineLoss: 9.79e-05
Beta: 2.25
CriticLoss: 0.726
KL: 0.00415
PolicyEntropy: 3.66
Steps: 750
TotalLoss: -0.191


on_policy_loss: -1.0099675762376137e-11. Off_policy_loss: -0.19607868194580078. Total Loss: -0.19607868195590045

***** Episode 360, Mean R = -47.3 *****
BaselineLoss: 0.000322
Beta: 2.25
CriticLoss: 0.745
KL: 0.002
PolicyEntropy: 3.65
Steps: 750
TotalLoss: -0.196


on_policy_loss: 1.604027251763303e-11. Off_policy_loss: -0.1959326171875. Total Loss: -0.1959326171714597

***** Episode 375, Mean R = -48.9 *****
BaselineLoss: 6.26e-05
Beta: 2.25
CriticLoss: 0.743
KL: 0.00345
PolicyEntropy: 3.64
Steps: 750
TotalLoss: -0.196


on_policy_loss: -4.185570704843636e-12. Off_policy_loss: -0.1958554267883301. Total Loss: -0.19585542679251566

***** Episode 390, Mean R = -47.1 *****
BaselineLoss: 0.000164
Beta: 2.25
CriticLoss: 0.742
KL: 0.00251
PolicyEntropy: 3.63
Steps: 750
TotalLoss: -0.196


on_policy_loss: -6.145900973327419e-12. Off_policy_loss: -0.1957217597961426. Total Loss: -0.19572175980228848

***** Episode 405, Mean R = -48.7 *****
BaselineLoss: 9.35e-05
Beta: 2.25
CriticLoss: 0.741
KL: 0.00371
PolicyEntropy: 3.62
Steps: 750
TotalLoss: -0.196


on_policy_loss: 3.814697298783661e-12. Off_policy_loss: -0.19557147979736328. Total Loss: -0.19557147979354858

***** Episode 420, Mean R = -48.8 *****
BaselineLoss: 6.94e-05
Beta: 2.25
CriticLoss: 0.741
KL: 0.00583
PolicyEntropy: 3.62
Steps: 750
TotalLoss: -0.196


on_policy_loss: 2.649095461511782e-12. Off_policy_loss: -0.19543386459350587. Total Loss: -0.19543386459085677

***** Episode 435, Mean R = -48.2 *****
BaselineLoss: 0.000117
Beta: 3.38
CriticLoss: 0.74
KL: 0.00724
PolicyEntropy: 3.63
Steps: 750
TotalLoss: -0.195


on_policy_loss: -4.005432193328791e-11. Off_policy_loss: -0.19535770416259765. Total Loss: -0.19535770420265197

***** Episode 450, Mean R = -48.0 *****
BaselineLoss: 0.000158
Beta: 2.25
CriticLoss: 0.74
KL: 0.00136
PolicyEntropy: 3.65
Steps: 750
TotalLoss: -0.195


on_policy_loss: 1.949734122301076e-11. Off_policy_loss: -0.19520278930664062. Total Loss: -0.19520278928714327

***** Episode 465, Mean R = -48.1 *****
BaselineLoss: 0.000135
Beta: 2.25
CriticLoss: 0.739
KL: 0.00524
PolicyEntropy: 3.64
Steps: 750
TotalLoss: -0.195


on_policy_loss: -2.5431316312089316e-12. Off_policy_loss: -0.1850769805908203. Total Loss: -0.18507698059336344

***** Episode 480, Mean R = -46.3 *****
BaselineLoss: 0.000386
Beta: 2.25
CriticLoss: 0.705
KL: 0.00349
PolicyEntropy: 3.63
Steps: 750
TotalLoss: -0.185


on_policy_loss: 1.6954210479980246e-11. Off_policy_loss: -0.19494632720947266. Total Loss: -0.19494632719251845

***** Episode 495, Mean R = -46.1 *****
BaselineLoss: 0.000208
Beta: 2.25
CriticLoss: 0.742
KL: 0.00547
PolicyEntropy: 3.62
Steps: 750
TotalLoss: -0.195


on_policy_loss: -4.2385526199950616e-12. Off_policy_loss: -0.19488906860351562. Total Loss: -0.19488906860775418

***** Episode 510, Mean R = -47.5 *****
BaselineLoss: 0.000147
Beta: 2.25
CriticLoss: 0.74
KL: 0.00233
PolicyEntropy: 3.62
Steps: 750
TotalLoss: -0.195


on_policy_loss: -1.1973911047865233e-11. Off_policy_loss: -0.19472827911376953. Total Loss: -0.19472827912574345

***** Episode 525, Mean R = -47.1 *****
BaselineLoss: 0.000213
Beta: 2.25
CriticLoss: 0.737
KL: 0.00556
PolicyEntropy: 3.62
Steps: 750
TotalLoss: -0.195


on_policy_loss: -2.935197566481899e-11. Off_policy_loss: -0.19461341857910155. Total Loss: -0.19461341860845352

***** Episode 540, Mean R = -46.0 *****
BaselineLoss: 0.000212
Beta: 2.25
CriticLoss: 0.736
KL: 0.00374
PolicyEntropy: 3.61
Steps: 750
TotalLoss: -0.195


on_policy_loss: -1.9073485901799358e-11. Off_policy_loss: -0.19454620361328126. Total Loss: -0.19454620363235475

***** Episode 555, Mean R = -39.6 *****
BaselineLoss: 0.000627
Beta: 2.25
CriticLoss: 0.735
KL: 0.00398
PolicyEntropy: 3.61
Steps: 750
TotalLoss: -0.195


on_policy_loss: -1.6106499837557446e-11. Off_policy_loss: -0.1944698715209961. Total Loss: -0.1944698715371026

***** Episode 570, Mean R = -46.4 *****
BaselineLoss: 0.000272
Beta: 2.25
CriticLoss: 0.735
KL: 0.00361
PolicyEntropy: 3.61
Steps: 750
TotalLoss: -0.194


on_policy_loss: -2.1192763099975308e-12. Off_policy_loss: -0.19446035385131835. Total Loss: -0.19446035385343763

***** Episode 585, Mean R = -44.7 *****
BaselineLoss: 0.000498
Beta: 2.25
CriticLoss: 0.734
KL: 0.00429
PolicyEntropy: 3.59
Steps: 750
TotalLoss: -0.194


on_policy_loss: 9.74867061150538e-12. Off_policy_loss: -0.19411785125732423. Total Loss: -0.19411785124757555

***** Episode 600, Mean R = -40.3 *****
BaselineLoss: 0.000722
Beta: 1.5
CriticLoss: 0.732
KL: 0.0014
PolicyEntropy: 3.59
Steps: 750
TotalLoss: -0.194


on_policy_loss: 1.706017371816415e-11. Off_policy_loss: -0.1840213966369629. Total Loss: -0.18402139661990272

***** Episode 615, Mean R = -43.9 *****
BaselineLoss: 0.000394
Beta: 2.25
CriticLoss: 0.699
KL: 0.00611
PolicyEntropy: 3.59
Steps: 750
TotalLoss: -0.184


on_policy_loss: 7.099575446053071e-12. Off_policy_loss: -0.19389574050903322. Total Loss: -0.19389574050193364

***** Episode 630, Mean R = -38.9 *****
BaselineLoss: 0.0007
Beta: 2.25
CriticLoss: 0.736
KL: 0.00317
PolicyEntropy: 3.56
Steps: 750
TotalLoss: -0.194


on_policy_loss: 1.064936346513908e-11. Off_policy_loss: -0.19376155853271484. Total Loss: -0.19376155852206547

***** Episode 645, Mean R = -39.5 *****
BaselineLoss: 0.000609
Beta: 2.25
CriticLoss: 0.733
KL: 0.00322
PolicyEntropy: 3.56
Steps: 750
TotalLoss: -0.194


on_policy_loss: 6.357828633933119e-12. Off_policy_loss: -0.1486572265625. Total Loss: -0.14865722655614216

***** Episode 660, Mean R = -40.0 *****
BaselineLoss: 0.000456
Beta: 2.25
CriticLoss: 0.577
KL: 0.00505
PolicyEntropy: 3.55
Steps: 750
TotalLoss: -0.149


on_policy_loss: 1.1854702108848868e-11. Off_policy_loss: -0.18358436584472657. Total Loss: -0.18358436583287185

***** Episode 675, Mean R = -40.8 *****
BaselineLoss: 0.0005
Beta: 2.25
CriticLoss: 0.713
KL: 0.00375
PolicyEntropy: 3.53
Steps: 750
TotalLoss: -0.184


on_policy_loss: 2.1192762359826625e-13. Off_policy_loss: -0.19344329833984375. Total Loss: -0.19344329833963184

***** Episode 690, Mean R = -37.7 *****
BaselineLoss: 0.00136
Beta: 2.25
CriticLoss: 0.742
KL: 0.00518
PolicyEntropy: 3.53
Steps: 750
TotalLoss: -0.193


on_policy_loss: -4.344516450297912e-12. Off_policy_loss: -0.11831701278686524. Total Loss: -0.11831701279120975

***** Episode 705, Mean R = -38.8 *****
BaselineLoss: 0.000574
Beta: 2.25
CriticLoss: 0.474
KL: 0.0052
PolicyEntropy: 3.53
Steps: 750
TotalLoss: -0.118


on_policy_loss: 4.159079480814398e-11. Off_policy_loss: -0.18324913024902345. Total Loss: -0.18324913020743266

***** Episode 720, Mean R = -36.3 *****
BaselineLoss: 0.000719
Beta: 2.25
CriticLoss: 0.725
KL: 0.00557
PolicyEntropy: 3.51
Steps: 750
TotalLoss: -0.183


on_policy_loss: -1.2609694029682334e-11. Off_policy_loss: -0.1881296730041504. Total Loss: -0.1881296730167601

***** Episode 735, Mean R = -31.7 *****
BaselineLoss: 0.000663
Beta: 2.25
CriticLoss: 0.731
KL: 0.006
PolicyEntropy: 3.5
Steps: 750
TotalLoss: -0.188


on_policy_loss: 1.8609894662328468e-11. Off_policy_loss: -0.07303732872009278. Total Loss: -0.07303732870148288

***** Episode 750, Mean R = -31.6 *****
BaselineLoss: 0.000624
Beta: 2.25
CriticLoss: 0.314
KL: 0.00592
PolicyEntropy: 3.53
Steps: 750
TotalLoss: -0.073


on_policy_loss: 1.0808309506652827e-11. Off_policy_loss: -0.17294490814208985. Total Loss: -0.17294490813128155

***** Episode 765, Mean R = -31.7 *****
BaselineLoss: 0.000552
Beta: 3.38
CriticLoss: 0.701
KL: 0.00707
PolicyEntropy: 3.53
Steps: 750
TotalLoss: -0.173


on_policy_loss: 1.9550324026340603e-11. Off_policy_loss: -0.19286258697509767. Total Loss: -0.19286258695554734

***** Episode 780, Mean R = -26.9 *****
BaselineLoss: 0.000726
Beta: 3.38
CriticLoss: 0.758
KL: 0.00233
PolicyEntropy: 3.5
Steps: 750
TotalLoss: -0.193


on_policy_loss: -6.357828633933119e-12. Off_policy_loss: -0.18778047561645508. Total Loss: -0.18778047562281291

***** Episode 795, Mean R = -26.5 *****
BaselineLoss: 0.00056
Beta: 3.38
CriticLoss: 0.716
KL: 0.003
PolicyEntropy: 3.5
Steps: 750
TotalLoss: -0.188


on_policy_loss: 4.927317220904115e-12. Off_policy_loss: -0.17264581680297852. Total Loss: -0.17264581679805122

***** Episode 810, Mean R = -23.3 *****
BaselineLoss: 0.00066
Beta: 3.38
CriticLoss: 0.655
KL: 0.0034
PolicyEntropy: 3.48
Steps: 750
TotalLoss: -0.173


on_policy_loss: 4.87433530575269e-12. Off_policy_loss: -0.19252819061279297. Total Loss: -0.19252819060791865

***** Episode 825, Mean R = -21.6 *****
BaselineLoss: 0.000587
Beta: 3.38
CriticLoss: 0.725
KL: 0.0027
PolicyEntropy: 3.46
Steps: 750
TotalLoss: -0.193


on_policy_loss: -1.3298458630591389e-11. Off_policy_loss: -0.18744752883911134. Total Loss: -0.18744752885240978

***** Episode 840, Mean R = -23.3 *****
BaselineLoss: 0.000494
Beta: 3.38
CriticLoss: 0.705
KL: 0.0027
PolicyEntropy: 3.45
Steps: 750
TotalLoss: -0.187


on_policy_loss: -5.298190923023564e-12. Off_policy_loss: -0.1473529529571533. Total Loss: -0.1473529529624515

***** Episode 855, Mean R = -19.3 *****
BaselineLoss: 0.000354
Beta: 3.38
CriticLoss: 0.57
KL: 0.00341
PolicyEntropy: 3.43
Steps: 750
TotalLoss: -0.147


on_policy_loss: -6.728702336052568e-12. Off_policy_loss: -0.18723159790039062. Total Loss: -0.18723159790711932

***** Episode 870, Mean R = -20.8 *****
BaselineLoss: 0.000377
Beta: 3.38
CriticLoss: 0.718
KL: 0.00323
PolicyEntropy: 3.41
Steps: 750
TotalLoss: -0.187


on_policy_loss: 1.356336791028904e-11. Off_policy_loss: -0.18715719223022462. Total Loss: -0.18715719221666124

***** Episode 885, Mean R = -17.1 *****
BaselineLoss: 0.000523
Beta: 3.38
CriticLoss: 0.71
KL: 0.00385
PolicyEntropy: 3.39
Steps: 750
TotalLoss: -0.187


on_policy_loss: 1.1867946625443436e-11. Off_policy_loss: -0.18698883056640625. Total Loss: -0.1869888305545383

***** Episode 900, Mean R = -15.3 *****
BaselineLoss: 0.000267
Beta: 3.38
CriticLoss: 0.702
KL: 0.00223
PolicyEntropy: 3.37
Steps: 750
TotalLoss: -0.187


on_policy_loss: 2.5431314535732476e-11. Off_policy_loss: -0.1819009780883789. Total Loss: -0.18190097806294758

***** Episode 915, Mean R = -14.3 *****
BaselineLoss: 0.000205
Beta: 3.38
CriticLoss: 0.684
KL: 0.00547
PolicyEntropy: 3.35
Steps: 750
TotalLoss: -0.182


on_policy_loss: -4.238552471965325e-13. Off_policy_loss: -0.1467863655090332. Total Loss: -0.14678636550945706

***** Episode 930, Mean R = -13.6 *****
BaselineLoss: 0.000306
Beta: 3.38
CriticLoss: 0.567
KL: 0.00336
PolicyEntropy: 3.33
Steps: 750
TotalLoss: -0.147


on_policy_loss: 1.1788473604686563e-11. Off_policy_loss: -0.1616627311706543. Total Loss: -0.1616627311588658

***** Episode 945, Mean R = -12.9 *****
BaselineLoss: 0.000215
Beta: 3.38
CriticLoss: 0.632
KL: 0.0037
PolicyEntropy: 3.32
Steps: 750
TotalLoss: -0.162


on_policy_loss: -2.1510653406645967e-11. Off_policy_loss: -0.18656442642211915. Total Loss: -0.1865644264436298

***** Episode 960, Mean R = -10.5 *****
BaselineLoss: 0.00019
Beta: 3.38
CriticLoss: 0.716
KL: 0.00321
PolicyEntropy: 3.3
Steps: 750
TotalLoss: -0.187


on_policy_loss: -3.814697298783661e-12. Off_policy_loss: -0.18145721435546874. Total Loss: -0.18145721435928344

***** Episode 975, Mean R = -10.0 *****
BaselineLoss: 9.94e-05
Beta: 3.38
CriticLoss: 0.686
KL: 0.00415
PolicyEntropy: 3.28
Steps: 750
TotalLoss: -0.181


on_policy_loss: 1.483493387392324e-11. Off_policy_loss: -0.19136344909667968. Total Loss: -0.19136344908184474

***** Episode 990, Mean R = -8.1 *****
BaselineLoss: 8.54e-05
Beta: 3.38
CriticLoss: 0.714
KL: 0.00405
PolicyEntropy: 3.26
Steps: 750
TotalLoss: -0.191


on_policy_loss: 1.780192112240305e-11. Off_policy_loss: -0.19122289657592773. Total Loss: -0.1912228965581258

***** Episode 1005, Mean R = -9.9 *****
BaselineLoss: 0.000187
Beta: 3.38
CriticLoss: 0.708
KL: 0.00423
PolicyEntropy: 3.22
Steps: 750
TotalLoss: -0.191


on_policy_loss: -4.662407645146989e-12. Off_policy_loss: -0.18111595153808593. Total Loss: -0.18111595154274834

***** Episode 1020, Mean R = -6.5 *****
BaselineLoss: 9.58e-05
Beta: 3.38
CriticLoss: 0.674
KL: 0.00356
PolicyEntropy: 3.22
Steps: 750
TotalLoss: -0.181


on_policy_loss: -7.41746693696162e-12. Off_policy_loss: -0.19100784301757812. Total Loss: -0.19100784302499557

***** Episode 1035, Mean R = -6.1 *****
BaselineLoss: 5.49e-05
Beta: 3.38
CriticLoss: 0.709
KL: 0.00391
PolicyEntropy: 3.2
Steps: 750
TotalLoss: -0.191


on_policy_loss: -1.2609694029682334e-11. Off_policy_loss: -0.18089746475219726. Total Loss: -0.18089746476480695

***** Episode 1050, Mean R = -6.9 *****
BaselineLoss: 7.1e-05
Beta: 3.38
CriticLoss: 0.674
KL: 0.00312
PolicyEntropy: 3.17
Steps: 750
TotalLoss: -0.181


on_policy_loss: -5.933973312721718e-12. Off_policy_loss: -0.1807823371887207. Total Loss: -0.18078233719465467

***** Episode 1065, Mean R = -5.4 *****
BaselineLoss: 4.28e-05
Beta: 3.38
CriticLoss: 0.675
KL: 0.00292
PolicyEntropy: 3.15
Steps: 750
TotalLoss: -0.181


on_policy_loss: 1.1550055726653833e-11. Off_policy_loss: -0.18566150665283204. Total Loss: -0.18566150664128198

***** Episode 1080, Mean R = -5.1 *****
BaselineLoss: 3.8e-05
Beta: 3.38
CriticLoss: 0.691
KL: 0.00356
PolicyEntropy: 3.13
Steps: 750
TotalLoss: -0.186


on_policy_loss: -1.6954210479980246e-11. Off_policy_loss: -0.18554943084716796. Total Loss: -0.18554943086412218

***** Episode 1095, Mean R = -4.4 *****
BaselineLoss: 4.52e-05
Beta: 3.38
CriticLoss: 0.689
KL: 0.00426
PolicyEntropy: 3.11
Steps: 750
TotalLoss: -0.186


on_policy_loss: 5.298190923023564e-12. Off_policy_loss: -0.17544372558593752. Total Loss: -0.17544372558063934

***** Episode 1110, Mean R = -4.9 *****
BaselineLoss: 4.56e-05
Beta: 3.38
CriticLoss: 0.655
KL: 0.00214
PolicyEntropy: 3.09
Steps: 750
TotalLoss: -0.175


on_policy_loss: 2.3523966774519065e-11. Off_policy_loss: -0.19032001495361328. Total Loss: -0.19032001493008932

***** Episode 1125, Mean R = -4.3 *****
BaselineLoss: 1.89e-05
Beta: 3.38
CriticLoss: 0.705
KL: 0.00285
PolicyEntropy: 3.06
Steps: 750
TotalLoss: -0.19


on_policy_loss: -5.086263262417863e-12. Off_policy_loss: -0.19021278381347656. Total Loss: -0.19021278381856283

***** Episode 1140, Mean R = -5.3 *****
BaselineLoss: 5.33e-05
Beta: 3.38
CriticLoss: 0.701
KL: 0.00263
PolicyEntropy: 3.03
Steps: 750
TotalLoss: -0.19


on_policy_loss: -4.662407645146989e-12. Off_policy_loss: -0.18010124206542968. Total Loss: -0.1801012420700921

***** Episode 1155, Mean R = -4.0 *****
BaselineLoss: 4.01e-05
Beta: 3.38
CriticLoss: 0.667
KL: 0.0038
PolicyEntropy: 3.02
Steps: 750
TotalLoss: -0.18


on_policy_loss: -8.477105239990123e-12. Off_policy_loss: -0.18499061584472656. Total Loss: -0.18499061585320367

***** Episode 1170, Mean R = -3.3 *****
BaselineLoss: 3.18e-05
Beta: 3.38
CriticLoss: 0.687
KL: 0.00427
PolicyEntropy: 2.99
Steps: 750
TotalLoss: -0.185


on_policy_loss: 4.2385526199950616e-12. Off_policy_loss: -0.16987445831298828. Total Loss: -0.16987445830874973

***** Episode 1185, Mean R = -3.7 *****
BaselineLoss: 2.26e-05
Beta: 3.38
CriticLoss: 0.638
KL: 0.00377
PolicyEntropy: 2.97
Steps: 750
TotalLoss: -0.17


on_policy_loss: 7.205539276355921e-12. Off_policy_loss: -0.18977115631103517. Total Loss: -0.18977115630382962

***** Episode 1200, Mean R = -3.2 *****
BaselineLoss: 1.73e-05
Beta: 3.38
CriticLoss: 0.707
KL: 0.003
PolicyEntropy: 2.94
Steps: 750
TotalLoss: -0.19


on_policy_loss: -2.829233854602838e-11. Off_policy_loss: -0.17965301513671875. Total Loss: -0.1796530151650111

***** Episode 1215, Mean R = -3.1 *****
BaselineLoss: 1.13e-05
Beta: 3.38
CriticLoss: 0.668
KL: 0.00314
PolicyEntropy: 2.92
Steps: 750
TotalLoss: -0.18


on_policy_loss: -7.947286088475873e-12. Off_policy_loss: -0.18454513549804688. Total Loss: -0.18454513550599416

***** Episode 1230, Mean R = -3.1 *****
BaselineLoss: 1.1e-05
Beta: 3.38
CriticLoss: 0.683
KL: 0.00352
PolicyEntropy: 2.92
Steps: 750
TotalLoss: -0.185


on_policy_loss: -3.28487814726941e-11. Off_policy_loss: -0.18944421768188477. Total Loss: -0.18944421771473355

***** Episode 1245, Mean R = -3.3 *****
BaselineLoss: 1.87e-05
Beta: 3.38
CriticLoss: 0.696
KL: 0.00498
PolicyEntropy: 2.9
Steps: 750
TotalLoss: -0.189


on_policy_loss: 1.780192112240305e-11. Off_policy_loss: -0.16432260513305663. Total Loss: -0.1643226051152547

***** Episode 1260, Mean R = -3.5 *****
BaselineLoss: 2.1e-05
Beta: 3.38
CriticLoss: 0.615
KL: 0.00478
PolicyEntropy: 2.87
Steps: 750
TotalLoss: -0.164


on_policy_loss: 8.900960561201525e-12. Off_policy_loss: -0.18421295166015625. Total Loss: -0.18421295165125529

***** Episode 1275, Mean R = -3.0 *****
BaselineLoss: 2.36e-05
Beta: 3.38
CriticLoss: 0.687
KL: 0.00465
PolicyEntropy: 2.85
Steps: 750
TotalLoss: -0.184


on_policy_loss: 5.828009482418869e-12. Off_policy_loss: -0.1841026496887207. Total Loss: -0.1841026496828927

***** Episode 1290, Mean R = -3.3 *****
BaselineLoss: 2.12e-05
Beta: 3.38
CriticLoss: 0.684
KL: 0.00593
PolicyEntropy: 2.83
Steps: 750
TotalLoss: -0.184


on_policy_loss: 1.6106499837557446e-11. Off_policy_loss: -0.1639980697631836. Total Loss: -0.1639980697470771

***** Episode 1305, Mean R = -2.8 *****
BaselineLoss: 8.91e-06
Beta: 3.38
CriticLoss: 0.618
KL: 0.00415
PolicyEntropy: 2.82
Steps: 750
TotalLoss: -0.164


on_policy_loss: -3.28487814726941e-12. Off_policy_loss: -0.1738871765136719. Total Loss: -0.17388717651695676

***** Episode 1320, Mean R = -3.2 *****
BaselineLoss: 1.92e-05
Beta: 3.38
CriticLoss: 0.654
KL: 0.00565
PolicyEntropy: 2.79
Steps: 750
TotalLoss: -0.174


on_policy_loss: 2.0133124796946808e-12. Off_policy_loss: -0.18377624511718751. Total Loss: -0.1837762451151742

***** Episode 1335, Mean R = -2.7 *****
BaselineLoss: 9.6e-06
Beta: 5.06
CriticLoss: 0.682
KL: 0.00604
PolicyEntropy: 2.76
Steps: 750
TotalLoss: -0.184


on_policy_loss: -1.7166138140585947e-11. Off_policy_loss: -0.16368297576904298. Total Loss: -0.1636829757862091

***** Episode 1350, Mean R = -2.1 *****
BaselineLoss: 6.29e-06
Beta: 5.06
CriticLoss: 0.616
KL: 0.00327
PolicyEntropy: 2.74
Steps: 750
TotalLoss: -0.164


on_policy_loss: -1.4093187653922238e-11. Off_policy_loss: -0.17856876373291017. Total Loss: -0.17856876374700337

***** Episode 1365, Mean R = -2.5 *****
BaselineLoss: 6.52e-06
Beta: 5.06
CriticLoss: 0.668
KL: 0.00224
PolicyEntropy: 2.72
Steps: 750
TotalLoss: -0.179


on_policy_loss: 7.735358427870172e-12. Off_policy_loss: -0.16846012115478515. Total Loss: -0.16846012114704978

***** Episode 1380, Mean R = -2.2 *****
BaselineLoss: 6.06e-06
Beta: 5.06
CriticLoss: 0.633
KL: 0.00288
PolicyEntropy: 2.71
Steps: 750
TotalLoss: -0.168


on_policy_loss: -2.5431316312089316e-12. Off_policy_loss: -0.17335445404052735. Total Loss: -0.17335445404307048

***** Episode 1395, Mean R = -2.3 *****
BaselineLoss: 8.06e-06
Beta: 5.06
CriticLoss: 0.65
KL: 0.00265
PolicyEntropy: 2.69
Steps: 750
TotalLoss: -0.173


on_policy_loss: 3.39084197757226e-12. Off_policy_loss: -0.17825237274169922. Total Loss: -0.17825237273830838

***** Episode 1410, Mean R = -2.5 *****
BaselineLoss: 6.29e-06
Beta: 5.06
CriticLoss: 0.665
KL: 0.00449
PolicyEntropy: 2.67
Steps: 750
TotalLoss: -0.178


on_policy_loss: 1.1020235983020635e-11. Off_policy_loss: -0.18813953399658204. Total Loss: -0.1881395339855618

***** Episode 1425, Mean R = -2.4 *****
BaselineLoss: 6.16e-06
Beta: 5.06
CriticLoss: 0.691
KL: 0.00296
PolicyEntropy: 2.66
Steps: 750
TotalLoss: -0.188


on_policy_loss: -8.930762499896143e-12. Off_policy_loss: -0.18303173065185546. Total Loss: -0.18303173066078623

***** Episode 1440, Mean R = -2.8 *****
BaselineLoss: 2.38e-05
Beta: 5.06
CriticLoss: 0.669
KL: 0.0022
PolicyEntropy: 2.64
Steps: 750
TotalLoss: -0.183


on_policy_loss: -3.39084197757226e-12. Off_policy_loss: -0.17792205810546877. Total Loss: -0.1779220581088596

***** Episode 1455, Mean R = -1.7 *****
BaselineLoss: 6.42e-06
Beta: 5.06
CriticLoss: 0.653
KL: 0.00272
PolicyEntropy: 2.63
Steps: 750
TotalLoss: -0.178


on_policy_loss: 9.324815290293978e-12. Off_policy_loss: -0.15281068801879882. Total Loss: -0.152810688009474

***** Episode 1470, Mean R = -2.1 *****
BaselineLoss: 5.28e-06
Beta: 5.06
CriticLoss: 0.577
KL: 0.00293
PolicyEntropy: 2.62
Steps: 750
TotalLoss: -0.153


on_policy_loss: 6.357828633933119e-12. Off_policy_loss: -0.14270074844360353. Total Loss: -0.1427007484372457

***** Episode 1485, Mean R = -1.9 *****
BaselineLoss: 4.29e-06
Beta: 5.06
CriticLoss: 0.555
KL: 0.00377
PolicyEntropy: 2.62
Steps: 750
TotalLoss: -0.143


on_policy_loss: -9.960598864230026e-12. Off_policy_loss: -0.1876076889038086. Total Loss: -0.1876076889137692

***** Episode 1500, Mean R = -1.9 *****
BaselineLoss: 3.4e-06
Beta: 5.06
CriticLoss: 0.707
KL: 0.00287
PolicyEntropy: 2.61
Steps: 750
TotalLoss: -0.188


on_policy_loss: -9.112887629688277e-12. Off_policy_loss: -0.17751140594482423. Total Loss: -0.1775114059539371

***** Episode 1515, Mean R = -2.4 *****
BaselineLoss: 6.38e-06
Beta: 5.06
CriticLoss: 0.659
KL: 0.00255
PolicyEntropy: 2.61
Steps: 750
TotalLoss: -0.178


on_policy_loss: 6.887647489387897e-13. Off_policy_loss: -0.14739948272705078. Total Loss: -0.14739948272636202

***** Episode 1530, Mean R = -2.6 *****
BaselineLoss: 8.04e-06
Beta: 5.06
CriticLoss: 0.559
KL: 0.00349
PolicyEntropy: 2.61
Steps: 750
TotalLoss: -0.147


on_policy_loss: 2.813339250451463e-11. Off_policy_loss: -0.07729511737823487. Total Loss: -0.07729511735010147

***** Episode 1545, Mean R = -2.4 *****
BaselineLoss: 8.25e-06
Beta: 5.06
CriticLoss: 0.336
KL: 0.00234
PolicyEntropy: 2.6
Steps: 750
TotalLoss: -0.0773


on_policy_loss: 5.351172542115516e-12. Off_policy_loss: -0.18220855712890624. Total Loss: -0.18220855712355508

***** Episode 1560, Mean R = -2.4 *****
BaselineLoss: 4.84e-06
Beta: 5.06
CriticLoss: 0.713
KL: 0.00223
PolicyEntropy: 2.58
Steps: 750
TotalLoss: -0.182


on_policy_loss: -1.377529557089474e-11. Off_policy_loss: -0.17212356567382814. Total Loss: -0.17212356568760342

***** Episode 1575, Mean R = -2.5 *****
BaselineLoss: 5.73e-06
Beta: 5.06
CriticLoss: 0.657
KL: 0.00294
PolicyEntropy: 2.57
Steps: 750
TotalLoss: -0.172


on_policy_loss: 3.655751553329386e-12. Off_policy_loss: -0.14703041076660156. Total Loss: -0.14703041076294582

***** Episode 1590, Mean R = -2.8 *****
BaselineLoss: 8.99e-06
Beta: 5.06
CriticLoss: 0.564
KL: 0.004
PolicyEntropy: 2.56
Steps: 750
TotalLoss: -0.147


on_policy_loss: 3.841188108329637e-12. Off_policy_loss: -0.1869309997558594. Total Loss: -0.1869309997520182

***** Episode 1605, Mean R = -2.3 *****
BaselineLoss: 6.97e-06
Beta: 3.38
CriticLoss: 0.693
KL: 0.00144
PolicyEntropy: 2.55
Steps: 750
TotalLoss: -0.187


on_policy_loss: -3.682242362875362e-12. Off_policy_loss: -0.16183156967163087. Total Loss: -0.16183156967531312

***** Episode 1620, Mean R = -2.1 *****
BaselineLoss: 5.36e-06
Beta: 3.38
CriticLoss: 0.604
KL: 0.00336
PolicyEntropy: 2.55
Steps: 750
TotalLoss: -0.162


on_policy_loss: -4.450480280600762e-12. Off_policy_loss: -0.16172050476074218. Total Loss: -0.16172050476519265

***** Episode 1635, Mean R = -2.3 *****
BaselineLoss: 4.79e-06
Beta: 3.38
CriticLoss: 0.605
KL: 0.0057
PolicyEntropy: 2.54
Steps: 750
TotalLoss: -0.162


on_policy_loss: -2.437167652876345e-12. Off_policy_loss: -0.16162097930908204. Total Loss: -0.1616209793115192

***** Episode 1650, Mean R = -2.6 *****
BaselineLoss: 6.44e-06
Beta: 3.38
CriticLoss: 0.608
KL: 0.00504
PolicyEntropy: 2.54
Steps: 750
TotalLoss: -0.162


on_policy_loss: -4.3657090979346924e-11. Off_policy_loss: -0.18651403427124025. Total Loss: -0.18651403431489733

***** Episode 1665, Mean R = -2.1 *****
BaselineLoss: 3.89e-06
Beta: 5.06
CriticLoss: 0.686
KL: 0.00794
PolicyEntropy: 2.51
Steps: 750
TotalLoss: -0.187


on_policy_loss: -1.3987223231500441e-11. Off_policy_loss: -0.1764232635498047. Total Loss: -0.17642326356379193

***** Episode 1680, Mean R = -1.9 *****
BaselineLoss: 3.16e-06
Beta: 5.06
CriticLoss: 0.644
KL: 0.00388
PolicyEntropy: 2.49
Steps: 750
TotalLoss: -0.176


on_policy_loss: 1.0596381549987654e-12. Off_policy_loss: -0.18129829406738282. Total Loss: -0.1812982940663232

***** Episode 1695, Mean R = -2.1 *****
BaselineLoss: 3.5e-06
Beta: 5.06
CriticLoss: 0.656
KL: 0.00378
PolicyEntropy: 2.49
Steps: 750
TotalLoss: -0.181


on_policy_loss: -8.47710494393065e-13. Off_policy_loss: -0.16119087219238282. Total Loss: -0.16119087219323053

***** Episode 1710, Mean R = -1.9 *****
BaselineLoss: 3e-06
Beta: 5.06
CriticLoss: 0.596
KL: 0.00297
PolicyEntropy: 2.48
Steps: 750
TotalLoss: -0.161


on_policy_loss: -7.41746693696162e-12. Off_policy_loss: -0.1760818862915039. Total Loss: -0.17608188629892135

***** Episode 1725, Mean R = -1.5 *****
BaselineLoss: 1.68e-06
Beta: 5.06
CriticLoss: 0.65
KL: 0.00267
PolicyEntropy: 2.47
Steps: 750
TotalLoss: -0.176


on_policy_loss: 5.298190774993827e-13. Off_policy_loss: -0.17598159790039064. Total Loss: -0.17598159789986081

***** Episode 1740, Mean R = -1.4 *****
BaselineLoss: 1.42e-06
Beta: 5.06
CriticLoss: 0.645
KL: 0.00243
PolicyEntropy: 2.45
Steps: 750
TotalLoss: -0.176


on_policy_loss: 7.205539276355921e-12. Off_policy_loss: -0.14587298393249512. Total Loss: -0.14587298392528958

***** Episode 1755, Mean R = -1.5 *****
BaselineLoss: 2.95e-06
Beta: 5.06
CriticLoss: 0.549
KL: 0.00203
PolicyEntropy: 2.44
Steps: 750
TotalLoss: -0.146


on_policy_loss: 3.39084197757226e-12. Off_policy_loss: -0.1757614517211914. Total Loss: -0.17576145171780055

***** Episode 1770, Mean R = -1.3 *****
BaselineLoss: 1.26e-06
Beta: 5.06
CriticLoss: 0.649
KL: 0.00324
PolicyEntropy: 2.44
Steps: 750
TotalLoss: -0.176


on_policy_loss: 7.099575446053071e-12. Off_policy_loss: -0.17066322326660158. Total Loss: -0.170663223259502

***** Episode 1785, Mean R = -2.1 *****
BaselineLoss: 5.87e-06
Beta: 5.06
CriticLoss: 0.627
KL: 0.00171
PolicyEntropy: 2.43
Steps: 750
TotalLoss: -0.171


on_policy_loss: 4.2385526199950616e-12. Off_policy_loss: -0.1605581283569336. Total Loss: -0.16055812835269503

***** Episode 1800, Mean R = -2.1 *****
BaselineLoss: 2.62e-06
Beta: 5.06
CriticLoss: 0.596
KL: 0.00365
PolicyEntropy: 2.43
Steps: 750
TotalLoss: -0.161


on_policy_loss: 1.8225775259376555e-11. Off_policy_loss: -0.1804446029663086. Total Loss: -0.18044460294808282

***** Episode 1815, Mean R = -2.1 *****
BaselineLoss: 2.75e-06
Beta: 5.06
CriticLoss: 0.66
KL: 0.00227
PolicyEntropy: 2.44
Steps: 750
TotalLoss: -0.18


on_policy_loss: -8.47710494393065e-13. Off_policy_loss: -0.10534334182739258. Total Loss: -0.10534334182824029

***** Episode 1830, Mean R = -1.7 *****
BaselineLoss: 1.26e-06
Beta: 5.06
CriticLoss: 0.423
KL: 0.00203
PolicyEntropy: 2.43
Steps: 750
TotalLoss: -0.105


on_policy_loss: 5.404154753326414e-12. Off_policy_loss: -0.11024229049682617. Total Loss: -0.11024229049142202

***** Episode 1845, Mean R = -1.7 *****
BaselineLoss: 2.17e-06
Beta: 5.06
CriticLoss: 0.456
KL: 0.00309
PolicyEntropy: 2.43
Steps: 750
TotalLoss: -0.11


on_policy_loss: -1.356336791028904e-11. Off_policy_loss: -0.17016616821289063. Total Loss: -0.170166168226454

***** Episode 1860, Mean R = -1.5 *****
BaselineLoss: 9.76e-07
Beta: 5.06
CriticLoss: 0.666
KL: 0.00291
PolicyEntropy: 2.43
Steps: 750
TotalLoss: -0.17


on_policy_loss: 2.3312038225734945e-12. Off_policy_loss: -0.17508165359497072. Total Loss: -0.17508165359263952

***** Episode 1875, Mean R = -1.9 *****
BaselineLoss: 2.45e-06
Beta: 5.06
CriticLoss: 0.656
KL: 0.0031
PolicyEntropy: 2.43
Steps: 750
TotalLoss: -0.175


on_policy_loss: 1.69542098878613e-12. Off_policy_loss: -0.12999164581298828. Total Loss: -0.12999164581129286

***** Episode 1890, Mean R = -1.9 *****
BaselineLoss: 2.23e-06
Beta: 5.06
CriticLoss: 0.502
KL: 0.00345
PolicyEntropy: 2.4
Steps: 750
TotalLoss: -0.13


on_policy_loss: 2.988179422421429e-11. Off_policy_loss: -0.16489143371582032. Total Loss: -0.16489143368593853

***** Episode 1905, Mean R = -1.7 *****
BaselineLoss: 2.36e-06
Beta: 5.06
CriticLoss: 0.622
KL: 0.00185
PolicyEntropy: 2.39
Steps: 750
TotalLoss: -0.165


on_policy_loss: 3.0517578390269286e-11. Off_policy_loss: -0.18479227066040038. Total Loss: -0.1847922706298828

***** Episode 1920, Mean R = -1.4 *****
BaselineLoss: 2.87e-06
Beta: 3.38
CriticLoss: 0.676
KL: 0.00136
PolicyEntropy: 2.39
Steps: 750
TotalLoss: -0.185


on_policy_loss: -3.41203474363283e-11. Off_policy_loss: -0.16469179153442384. Total Loss: -0.1646917915685442

***** Episode 1935, Mean R = -2.0 *****
BaselineLoss: 7.42e-06
Beta: 3.38
CriticLoss: 0.602
KL: 0.00254
PolicyEntropy: 2.36
Steps: 750
TotalLoss: -0.165


on_policy_loss: 5.298190923023564e-12. Off_policy_loss: -0.16458499908447266. Total Loss: -0.16458499907917448

***** Episode 1950, Mean R = -1.8 *****
BaselineLoss: 3.3e-06
Beta: 3.38
CriticLoss: 0.604
KL: 0.00555
PolicyEntropy: 2.36
Steps: 750
TotalLoss: -0.165


on_policy_loss: -5.933973312721718e-12. Off_policy_loss: -0.16947946548461915. Total Loss: -0.16947946549055312

***** Episode 1965, Mean R = -1.7 *****
BaselineLoss: 1.3e-06
Beta: 3.38
CriticLoss: 0.621
KL: 0.00578
PolicyEntropy: 2.37
Steps: 750
TotalLoss: -0.169


on_policy_loss: 6.1988831845383175e-12. Off_policy_loss: -0.15437202453613283. Total Loss: -0.15437202452993395

***** Episode 1980, Mean R = -2.3 *****
BaselineLoss: 6.11e-06
Beta: 5.06
CriticLoss: 0.573
KL: 0.00631
PolicyEntropy: 2.36
Steps: 750
TotalLoss: -0.154


on_policy_loss: 2.649095461511782e-12. Off_policy_loss: -0.1792717170715332. Total Loss: -0.1792717170688841

***** Episode 1995, Mean R = -2.1 *****
BaselineLoss: 2.79e-06
Beta: 5.06
CriticLoss: 0.65
KL: 0.00548
PolicyEntropy: 2.35
Steps: 750
TotalLoss: -0.179


on_policy_loss: 2.670288049936668e-11. Off_policy_loss: -0.14916275978088378. Total Loss: -0.1491627597541809

***** Episode 2010, Mean R = -2.1 *****
BaselineLoss: 3.31e-06
Beta: 5.06
CriticLoss: 0.556
KL: 0.00236
PolicyEntropy: 2.35
Steps: 750
TotalLoss: -0.149


on_policy_loss: -2.1192762359826625e-13. Off_policy_loss: -0.1840599060058594. Total Loss: -0.1840599060060713

***** Episode 2025, Mean R = -1.7 *****
BaselineLoss: 1.7e-06
Beta: 5.06
CriticLoss: 0.665
KL: 0.00215
PolicyEntropy: 2.35
Steps: 750
TotalLoss: -0.184


on_policy_loss: -4.87433530575269e-12. Off_policy_loss: -0.16895431518554688. Total Loss: -0.1689543151904212

***** Episode 2040, Mean R = -1.5 *****
BaselineLoss: 2.71e-06
Beta: 5.06
CriticLoss: 0.612
KL: 0.0034
PolicyEntropy: 2.34
Steps: 750
TotalLoss: -0.169


on_policy_loss: 7.205539276355921e-12. Off_policy_loss: -0.16884876251220704. Total Loss: -0.1688487625050015

***** Episode 2055, Mean R = -2.3 *****
BaselineLoss: 3.7e-06
Beta: 5.06
CriticLoss: 0.609
KL: 0.00255
PolicyEntropy: 2.34
Steps: 750
TotalLoss: -0.169


on_policy_loss: 8.689032900595824e-12. Off_policy_loss: -0.15873650550842286. Total Loss: -0.15873650549973384

***** Episode 2070, Mean R = -1.7 *****
BaselineLoss: 1.67e-06
Beta: 5.06
CriticLoss: 0.582
KL: 0.00511
PolicyEntropy: 2.35
Steps: 750
TotalLoss: -0.159


on_policy_loss: -1.313951258907764e-11. Off_policy_loss: -0.1536301040649414. Total Loss: -0.15363010407808092

***** Episode 2085, Mean R = -2.1 *****
BaselineLoss: 1.9e-06
Beta: 5.06
CriticLoss: 0.571
KL: 0.00532
PolicyEntropy: 2.35
Steps: 750
TotalLoss: -0.154


on_policy_loss: 8.900960561201525e-12. Off_policy_loss: -0.16353153228759765. Total Loss: -0.16353153227869668

***** Episode 2100, Mean R = -1.6 *****
BaselineLoss: 4.96e-06
Beta: 5.06
CriticLoss: 0.603
KL: 0.00178
PolicyEntropy: 2.33
Steps: 750
TotalLoss: -0.164


on_policy_loss: -5.4571363724183655e-12. Off_policy_loss: -0.14342572212219237. Total Loss: -0.1434257221276495

***** Episode 2115, Mean R = -1.8 *****
BaselineLoss: 3e-06
Beta: 5.06
CriticLoss: 0.54
KL: 0.00436
PolicyEntropy: 2.32
Steps: 750
TotalLoss: -0.143


on_policy_loss: -1.758999346179735e-11. Off_policy_loss: -0.16332422256469728. Total Loss: -0.16332422258228727

***** Episode 2130, Mean R = -2.2 *****
BaselineLoss: 3.57e-06
Beta: 5.06
CriticLoss: 0.606
KL: 0.00206
PolicyEntropy: 2.32
Steps: 750
TotalLoss: -0.163


on_policy_loss: -1.9073486493918304e-12. Off_policy_loss: -0.17822717666625976. Total Loss: -0.1782271766681671

***** Episode 2145, Mean R = -2.0 *****
BaselineLoss: 3.49e-06
Beta: 5.06
CriticLoss: 0.645
KL: 0.00252
PolicyEntropy: 2.31
Steps: 750
TotalLoss: -0.178


on_policy_loss: -1.377529557089474e-11. Off_policy_loss: -0.1581219959259033. Total Loss: -0.1581219959396786

***** Episode 2160, Mean R = -1.5 *****
BaselineLoss: 2.68e-06
Beta: 5.06
CriticLoss: 0.576
KL: 0.00304
PolicyEntropy: 2.32
Steps: 750
TotalLoss: -0.158


on_policy_loss: 8.477105239990123e-12. Off_policy_loss: -0.17801715850830077. Total Loss: -0.17801715849982366

***** Episode 2175, Mean R = -1.4 *****
BaselineLoss: 1.99e-06
Beta: 5.06
CriticLoss: 0.636
KL: 0.00341
PolicyEntropy: 2.33
Steps: 750
TotalLoss: -0.178


on_policy_loss: 3.25308893896666e-11. Off_policy_loss: -0.17790933609008788. Total Loss: -0.177909336057557

***** Episode 2190, Mean R = -1.9 *****
BaselineLoss: 2.45e-06
Beta: 5.06
CriticLoss: 0.631
KL: 0.00285
PolicyEntropy: 2.33
Steps: 750
TotalLoss: -0.178


on_policy_loss: -7.735358427870172e-12. Off_policy_loss: -0.18279956817626952. Total Loss: -0.1827995681840049

***** Episode 2205, Mean R = -2.0 *****
BaselineLoss: 3.87e-06
Beta: 5.06
CriticLoss: 0.64
KL: 0.00214
PolicyEntropy: 2.33
Steps: 750
TotalLoss: -0.183


on_policy_loss: 3.827942703556649e-12. Off_policy_loss: -0.17268362045288085. Total Loss: -0.17268362044905292

***** Episode 2220, Mean R = -1.9 *****
BaselineLoss: 3.4e-06
Beta: 5.06
CriticLoss: 0.609
KL: 0.00212
PolicyEntropy: 2.33
Steps: 750
TotalLoss: -0.173


on_policy_loss: -1.8437702919982256e-11. Off_policy_loss: -0.16756841659545899. Total Loss: -0.16756841661389668

***** Episode 2235, Mean R = -1.9 *****
BaselineLoss: 2.3e-06
Beta: 5.06
CriticLoss: 0.595
KL: 0.00315
PolicyEntropy: 2.32
Steps: 750
TotalLoss: -0.168


on_policy_loss: 2.5431316312089316e-12. Off_policy_loss: -0.1674581527709961. Total Loss: -0.16745815276845297

***** Episode 2250, Mean R = -2.1 *****
BaselineLoss: 3.13e-06
Beta: 5.06
CriticLoss: 0.599
KL: 0.00308
PolicyEntropy: 2.33
Steps: 750
TotalLoss: -0.167


on_policy_loss: -1.5894572695055822e-13. Off_policy_loss: -0.17234756469726562. Total Loss: -0.17234756469742457

***** Episode 2265, Mean R = -1.3 *****
BaselineLoss: 1.65e-06
Beta: 5.06
CriticLoss: 0.615
KL: 0.00233
PolicyEntropy: 2.32
Steps: 750
TotalLoss: -0.172


on_policy_loss: -1.0596381549987654e-12. Off_policy_loss: -0.1722380065917969. Total Loss: -0.1722380065928565

***** Episode 2280, Mean R = -2.0 *****
BaselineLoss: 1.27e-05
Beta: 5.06
CriticLoss: 0.611
KL: 0.00181
PolicyEntropy: 2.29
Steps: 750
TotalLoss: -0.172


on_policy_loss: -1.4411078552711841e-11. Off_policy_loss: -0.16212799072265624. Total Loss: -0.16212799073706732

***** Episode 2295, Mean R = -1.4 *****
BaselineLoss: 3.93e-06
Beta: 5.06
CriticLoss: 0.584
KL: 0.00181
PolicyEntropy: 2.29
Steps: 750
TotalLoss: -0.162


on_policy_loss: -3.7723116482387314e-11. Off_policy_loss: -0.17201999664306641. Total Loss: -0.17201999668078954

***** Episode 2310, Mean R = -2.0 *****
BaselineLoss: 2.54e-06
Beta: 5.06
CriticLoss: 0.615
KL: 0.00255
PolicyEntropy: 2.3
Steps: 750
TotalLoss: -0.172


on_policy_loss: -5.404154753326414e-12. Off_policy_loss: -0.1619158935546875. Total Loss: -0.16191589356009164

***** Episode 2325, Mean R = -1.9 *****
BaselineLoss: 9.72e-07
Beta: 5.06
CriticLoss: 0.584
KL: 0.00304
PolicyEntropy: 2.32
Steps: 750
TotalLoss: -0.162


on_policy_loss: -1.1973911047865233e-11. Off_policy_loss: -0.17180706024169923. Total Loss: -0.17180706025367315

***** Episode 2340, Mean R = -2.2 *****
BaselineLoss: 2e-06
Beta: 5.06
CriticLoss: 0.614
KL: 0.00231
PolicyEntropy: 2.32
Steps: 750
TotalLoss: -0.172


on_policy_loss: 8.053249918778723e-12. Off_policy_loss: -0.1516999340057373. Total Loss: -0.15169993399768406

***** Episode 2355, Mean R = -1.7 *****
BaselineLoss: 2.66e-06
Beta: 3.38
CriticLoss: 0.552
KL: 0.00143
PolicyEntropy: 2.3
Steps: 750
TotalLoss: -0.152


on_policy_loss: -1.483493387392324e-11. Off_policy_loss: -0.17159358978271486. Total Loss: -0.1715935897975498

***** Episode 2370, Mean R = -2.0 *****
BaselineLoss: 5.64e-06
Beta: 3.38
CriticLoss: 0.616
KL: 0.0045
PolicyEntropy: 2.28
Steps: 750
TotalLoss: -0.172


on_policy_loss: -4.6200222906615334e-11. Off_policy_loss: -0.17148849487304688. Total Loss: -0.1714884949192471

***** Episode 2385, Mean R = -2.3 *****
BaselineLoss: 4.29e-06
Beta: 5.06
CriticLoss: 0.612
KL: 0.00715
PolicyEntropy: 2.27
Steps: 750
TotalLoss: -0.171


on_policy_loss: -9.74867061150538e-12. Off_policy_loss: -0.1713827133178711. Total Loss: -0.17138271332761978

***** Episode 2400, Mean R = -1.6 *****
BaselineLoss: 3.55e-06
Beta: 5.06
CriticLoss: 0.606
KL: 0.00238
PolicyEntropy: 2.27
Steps: 750
TotalLoss: -0.171


on_policy_loss: 1.4119677871349267e-11. Off_policy_loss: -0.14627179145812988. Total Loss: -0.1462717914440102

***** Episode 2415, Mean R = -2.1 *****
BaselineLoss: 3.85e-06
Beta: 5.06
CriticLoss: 0.535
KL: 0.00286
PolicyEntropy: 2.25
Steps: 750
TotalLoss: -0.146


on_policy_loss: 9.74867061150538e-12. Off_policy_loss: -0.1611666488647461. Total Loss: -0.16116664885499743

***** Episode 2430, Mean R = -1.9 *****
BaselineLoss: 4.73e-06
Beta: 5.06
CriticLoss: 0.587
KL: 0.00279
PolicyEntropy: 2.23
Steps: 750
TotalLoss: -0.161


on_policy_loss: 7.41746693696162e-12. Off_policy_loss: -0.17106643676757813. Total Loss: -0.17106643676016067

***** Episode 2445, Mean R = -1.8 *****
BaselineLoss: 2.44e-06
Beta: 5.06
CriticLoss: 0.614
KL: 0.00236
PolicyEntropy: 2.22
Steps: 750
TotalLoss: -0.171


on_policy_loss: -1.8437702919982256e-11. Off_policy_loss: -0.1559623146057129. Total Loss: -0.15596231462415058

***** Episode 2460, Mean R = -2.3 *****
BaselineLoss: 4.1e-06
Beta: 5.06
CriticLoss: 0.564
KL: 0.00326
PolicyEntropy: 2.21
Steps: 750
TotalLoss: -0.156


on_policy_loss: -2.7550589957551587e-12. Off_policy_loss: -0.17085695266723633. Total Loss: -0.17085695266999137

***** Episode 2475, Mean R = -2.2 *****
BaselineLoss: 2.32e-06
Beta: 5.06
CriticLoss: 0.608
KL: 0.00272
PolicyEntropy: 2.19
Steps: 750
TotalLoss: -0.171


on_policy_loss: 2.0159615606492782e-11. Off_policy_loss: -0.17575271606445314. Total Loss: -0.17575271604429352

***** Episode 2490, Mean R = -1.7 *****
BaselineLoss: 1.16e-06
Beta: 5.06
CriticLoss: 0.619
KL: 0.0027
PolicyEntropy: 2.18
Steps: 750
TotalLoss: -0.176


on_policy_loss: -2.966986656360859e-12. Off_policy_loss: -0.1556451416015625. Total Loss: -0.1556451416045295

***** Episode 2505, Mean R = -2.3 *****
BaselineLoss: 1.97e-06
Beta: 5.06
CriticLoss: 0.558
KL: 0.00371
PolicyEntropy: 2.19
Steps: 750
TotalLoss: -0.156


on_policy_loss: 4.662407645146989e-12. Off_policy_loss: -0.1555354881286621. Total Loss: -0.1555354881239997

***** Episode 2520, Mean R = -2.3 *****
BaselineLoss: 3.37e-06
Beta: 5.06
CriticLoss: 0.562
KL: 0.00381
PolicyEntropy: 2.18
Steps: 750
TotalLoss: -0.156


on_policy_loss: 2.214643757270096e-11. Off_policy_loss: -0.18043025970458984. Total Loss: -0.1804302596824434

***** Episode 2535, Mean R = -1.6 *****
BaselineLoss: 2.26e-06
Beta: 5.06
CriticLoss: 0.634
KL: 0.00253
PolicyEntropy: 2.19
Steps: 750
TotalLoss: -0.18


on_policy_loss: 4.0531157689353375e-12. Off_policy_loss: -0.17032312393188476. Total Loss: -0.17032312392783164

***** Episode 2550, Mean R = -1.7 *****
BaselineLoss: 4.73e-06
Beta: 5.06
CriticLoss: 0.598
KL: 0.00239
PolicyEntropy: 2.18
Steps: 750
TotalLoss: -0.17


on_policy_loss: 7.523430767264471e-12. Off_policy_loss: -0.17521635055541993. Total Loss: -0.1752163505478965

***** Episode 2565, Mean R = -1.7 *****
BaselineLoss: 2.9e-06
Beta: 5.06
CriticLoss: 0.609
KL: 0.00153
PolicyEntropy: 2.17
Steps: 750
TotalLoss: -0.175


on_policy_loss: -1.0490417423625331e-11. Off_policy_loss: -0.17010616302490233. Total Loss: -0.17010616303539275

***** Episode 2580, Mean R = -1.8 *****
BaselineLoss: 2.43e-06
Beta: 5.06
CriticLoss: 0.594
KL: 0.00232
PolicyEntropy: 2.16
Steps: 750
TotalLoss: -0.17


on_policy_loss: -2.0345053049671453e-11. Off_policy_loss: -0.1599920654296875. Total Loss: -0.15999206545003256

***** Episode 2595, Mean R = -2.2 *****
BaselineLoss: 3.48e-06
Beta: 5.06
CriticLoss: 0.567
KL: 0.00244
PolicyEntropy: 2.16
Steps: 750
TotalLoss: -0.16


on_policy_loss: 3.39084197757226e-12. Off_policy_loss: -0.15488327980041505. Total Loss: -0.1548832797970242

***** Episode 2610, Mean R = -1.7 *****
BaselineLoss: 2.57e-06
Beta: 5.06
CriticLoss: 0.558
KL: 0.00421
PolicyEntropy: 2.17
Steps: 750
TotalLoss: -0.155


on_policy_loss: -2.225240140300381e-12. Off_policy_loss: -0.1547792434692383. Total Loss: -0.15477924347146355

***** Episode 2625, Mean R = -1.9 *****
BaselineLoss: 3.47e-06
Beta: 5.06
CriticLoss: 0.561
KL: 0.00191
PolicyEntropy: 2.16
Steps: 750
TotalLoss: -0.155


on_policy_loss: -1.0596381549987654e-12. Off_policy_loss: -0.15467702865600585. Total Loss: -0.15467702865706548

***** Episode 2640, Mean R = -1.9 *****
BaselineLoss: 3.73e-06
Beta: 5.06
CriticLoss: 0.562
KL: 0.00409
PolicyEntropy: 2.15
Steps: 750
TotalLoss: -0.155


on_policy_loss: 6.56975629453882e-12. Off_policy_loss: -0.13957768440246582. Total Loss: -0.13957768439589607

***** Episode 2655, Mean R = -2.1 *****
BaselineLoss: 2.66e-06
Beta: 5.06
CriticLoss: 0.521
KL: 0.00195
PolicyEntropy: 2.16
Steps: 750
TotalLoss: -0.14


on_policy_loss: 1.1656018964837737e-11. Off_policy_loss: -0.1794784927368164. Total Loss: -0.1794784927251604

***** Episode 2670, Mean R = -2.2 *****
BaselineLoss: 2.61e-06
Beta: 5.06
CriticLoss: 0.638
KL: 0.00385
PolicyEntropy: 2.17
Steps: 750
TotalLoss: -0.179


on_policy_loss: 2.1616619013305654e-11. Off_policy_loss: -0.17437976837158203. Total Loss: -0.1743797683499654

***** Episode 2685, Mean R = -1.7 *****
BaselineLoss: 2.9e-06
Beta: 5.06
CriticLoss: 0.608
KL: 0.00337
PolicyEntropy: 2.17
Steps: 750
TotalLoss: -0.174


on_policy_loss: 5.1922270927207136e-12. Off_policy_loss: -0.17426977157592774. Total Loss: -0.1742697715707355

***** Episode 2700, Mean R = -1.5 *****
BaselineLoss: 1.44e-06
Beta: 5.06
CriticLoss: 0.6
KL: 0.00204
PolicyEntropy: 2.18
Steps: 750
TotalLoss: -0.174


on_policy_loss: -4.238552471965325e-13. Off_policy_loss: -0.17415815353393554. Total Loss: -0.1741581535343594

***** Episode 2715, Mean R = -1.8 *****
BaselineLoss: 2.82e-06
Beta: 5.06
CriticLoss: 0.599
KL: 0.00157
PolicyEntropy: 2.19
Steps: 750
TotalLoss: -0.174


on_policy_loss: 7.576412386356423e-12. Off_policy_loss: -0.16904521942138673. Total Loss: -0.16904521941381032

***** Episode 2730, Mean R = -2.0 *****
BaselineLoss: 7.95e-06
Beta: 5.06
CriticLoss: 0.586
KL: 0.00248
PolicyEntropy: 2.17
Steps: 750
TotalLoss: -0.169


on_policy_loss: 7.046593826961119e-12. Off_policy_loss: -0.15393397331237793. Total Loss: -0.15393397330533135

***** Episode 2745, Mean R = -1.7 *****
BaselineLoss: 2.17e-06
Beta: 5.06
CriticLoss: 0.547
KL: 0.00305
PolicyEntropy: 2.17
Steps: 750
TotalLoss: -0.154


on_policy_loss: -1.9073485901799358e-11. Off_policy_loss: -0.17882675170898438. Total Loss: -0.17882675172805787

***** Episode 2760, Mean R = -1.7 *****
BaselineLoss: 2.11e-06
Beta: 5.06
CriticLoss: 0.619
KL: 0.00173
PolicyEntropy: 2.14
Steps: 750
TotalLoss: -0.179


on_policy_loss: -2.0133124796946808e-12. Off_policy_loss: -0.16372085571289063. Total Loss: -0.16372085571490394

***** Episode 2775, Mean R = -2.1 *****
BaselineLoss: 2.92e-06
Beta: 5.06
CriticLoss: 0.572
KL: 0.00288
PolicyEntropy: 2.11
Steps: 750
TotalLoss: -0.164


on_policy_loss: 7.205539276355921e-12. Off_policy_loss: -0.1586078643798828. Total Loss: -0.15860786437267727

***** Episode 2790, Mean R = -1.9 *****
BaselineLoss: 3.37e-06
Beta: 5.06
CriticLoss: 0.56
KL: 0.00407
PolicyEntropy: 2.12
Steps: 750
TotalLoss: -0.159


on_policy_loss: -3.39084197757226e-12. Off_policy_loss: -0.15850220680236818. Total Loss: -0.15850220680575902

***** Episode 2805, Mean R = -1.9 *****
BaselineLoss: 3.45e-06
Beta: 5.06
CriticLoss: 0.565
KL: 0.0029
PolicyEntropy: 2.13
Steps: 750
TotalLoss: -0.159


on_policy_loss: 5.086263262417863e-12. Off_policy_loss: -0.14839900016784668. Total Loss: -0.14839900016276042

***** Episode 2820, Mean R = -1.7 *****
BaselineLoss: 1.91e-06
Beta: 5.06
CriticLoss: 0.538
KL: 0.00273
PolicyEntropy: 2.13
Steps: 750
TotalLoss: -0.148


on_policy_loss: 6.887647489387897e-13. Off_policy_loss: -0.13829756736755372. Total Loss: -0.13829756736686497

***** Episode 2835, Mean R = -2.3 *****
BaselineLoss: 7.24e-06
Beta: 5.06
CriticLoss: 0.513
KL: 0.00282
PolicyEntropy: 2.12
Steps: 750
TotalLoss: -0.138


on_policy_loss: -2.0159615606492782e-11. Off_policy_loss: -0.15819910049438476. Total Loss: -0.1581991005145444

***** Episode 2850, Mean R = -2.0 *****
BaselineLoss: 4.93e-06
Beta: 5.06
CriticLoss: 0.574
KL: 0.00226
PolicyEntropy: 2.13
Steps: 750
TotalLoss: -0.158


on_policy_loss: 1.1497074107561882e-11. Off_policy_loss: -0.12810383796691896. Total Loss: -0.12810383795542188

***** Episode 2865, Mean R = -2.4 *****
BaselineLoss: 1.07e-05
Beta: 5.06
CriticLoss: 0.484
KL: 0.00394
PolicyEntropy: 2.13
Steps: 750
TotalLoss: -0.128


on_policy_loss: -1.5523698474832297e-11. Off_policy_loss: -0.1730085563659668. Total Loss: -0.17300855638149049

***** Episode 2880, Mean R = -2.1 *****
BaselineLoss: 4.18e-06
Beta: 5.06
CriticLoss: 0.612
KL: 0.00325
PolicyEntropy: 2.12
Steps: 750
TotalLoss: -0.173


on_policy_loss: -2.7974446463000883e-11. Off_policy_loss: -0.13291008949279787. Total Loss: -0.13291008952077232

***** Episode 2895, Mean R = -1.7 *****
BaselineLoss: 1.63e-06
Beta: 5.06
CriticLoss: 0.493
KL: 0.00381
PolicyEntropy: 2.11
Steps: 750
TotalLoss: -0.133


on_policy_loss: 7.629394597567321e-12. Off_policy_loss: -0.17280887603759765. Total Loss: -0.17280887602996825

***** Episode 2910, Mean R = -1.9 *****
BaselineLoss: 1.38e-06
Beta: 5.06
CriticLoss: 0.606
KL: 0.00202
PolicyEntropy: 2.1
Steps: 750
TotalLoss: -0.173


on_policy_loss: -8.47710494393065e-13. Off_policy_loss: -0.17770832061767577. Total Loss: -0.17770832061852349

***** Episode 2925, Mean R = -1.9 *****
BaselineLoss: 1.49e-06
Beta: 5.06
CriticLoss: 0.61
KL: 0.00567
PolicyEntropy: 2.1
Steps: 750
TotalLoss: -0.178


on_policy_loss: 1.1232163643626336e-11. Off_policy_loss: -0.15760028839111329. Total Loss: -0.15760028837988113

***** Episode 2940, Mean R = -2.0 *****
BaselineLoss: 2.96e-06
Beta: 5.06
CriticLoss: 0.549
KL: 0.00265
PolicyEntropy: 2.1
Steps: 750
TotalLoss: -0.158


on_policy_loss: -4.5034621957521876e-12. Off_policy_loss: -0.14249028205871583. Total Loss: -0.1424902820632193

***** Episode 2955, Mean R = -1.7 *****
BaselineLoss: 1.11e-06
Beta: 5.06
CriticLoss: 0.514
KL: 0.00342
PolicyEntropy: 2.1
Steps: 750
TotalLoss: -0.142


on_policy_loss: 1.0702345084231032e-11. Off_policy_loss: -0.16738576889038087. Total Loss: -0.16738576887967851

***** Episode 2970, Mean R = -2.3 *****
BaselineLoss: 4.48e-06
Beta: 5.06
CriticLoss: 0.588
KL: 0.00418
PolicyEntropy: 2.09
Steps: 750
TotalLoss: -0.167


on_policy_loss: -2.702077376663207e-12. Off_policy_loss: -0.14228355407714843. Total Loss: -0.14228355407985052

***** Episode 2985, Mean R = -2.1 *****
BaselineLoss: 6.04e-06
Beta: 5.06
CriticLoss: 0.516
KL: 0.00187
PolicyEntropy: 2.08
Steps: 750
TotalLoss: -0.142


on_policy_loss: -1.0808309506652827e-11. Off_policy_loss: -0.16718097686767577. Total Loss: -0.16718097687848407

***** Episode 3000, Mean R = -1.7 *****
BaselineLoss: 3.53e-06
Beta: 5.06
CriticLoss: 0.585
KL: 0.00298
PolicyEntropy: 2.07
Steps: 750
TotalLoss: -0.167