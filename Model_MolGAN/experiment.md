## 0.5 0.5 discrete

train: 0.5 x tree Reward(0 or 1) + 0.5 x connectivity Reward(0 or 1)

evaluation: 1-abs((size-(v-1)) / (v-1)), 1/Part Number

best iteration: 20000

average reward: 0.19

![image-20200823203140569](experiment.assets/image-20200823203140569.png)

![image-20200823203154447](experiment.assets/image-20200823203154447.png)



## 0.5 0.5 continuous

train: 0.5 x tree Reward(continuous) + 0.5 x connectivity Reward(continuous)

evaluation: 1-abs((size-(v-1)) / (v-1)), 1/Part Number

best iteration: 120000

average reward: 0.28

![image-20200823235701036](experiment.assets/image-20200823235701036.png)

![image-20200823235541958](experiment.assets/image-20200823235541958.png)

## 0.7 0.3 discrete

train: 0.7 x tree Reward(0 or 1) + 0.3 x connectivity Reward(0 or 1)

evaluation: 1-abs((size-(v-1)) / (v-1)), 1/Part Number

best iteration: 60000

average reward: 0.20

![image-20200824100926689](experiment.assets/image-20200824100926689.png)

![image-20200824100910108](experiment.assets/image-20200824100910108.png)



## 0.7 0.3 continuousquarantine

train: 0.7 x tree Reward(continuous) + 0.3 x connectivity Reward(continuous)

evaluation: 1-abs((size-(v-1)) / (v-1)), 1/Part Number

best iteration: 20000

average reward: 0.09

![image-20200824160755735](experiment.assets/image-20200824160755735.png)

![image-20200824160730273](experiment.assets/image-20200824160730273.png)

## 0.3 0.7 discrete

train: 0.3 x tree Reward(0 or 1) + 0.7 x connectivity Reward(0 or 1)

evaluation: 1-abs((size-(v-1)) / (v-1)), 1/Part Number

best iteration: 120000

average reward: 0.27

![image-20200824195626208](experiment.assets/image-20200824195626208.png)

![image-20200824195715268](experiment.assets/image-20200824195715268.png)