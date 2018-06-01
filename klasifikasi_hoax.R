library(caret)
library(tm)
library(SnowballC)
library(arm)
# Training data.
data <- c('Video Porno Aryodj Ini Kata MKD DPR',
          'Jokowi Atur Waktu Temui Amien Rais',
          'Fadli Zon Akui Prabowo dan Puan Pernah Bertemu',
          'Dibully Netizen Megawati Angkat Bicara',
          'Zidane tinggalkan Madrid',
          'TommyLimmm bakal keluar dari Tim2One',
          'Zlatan keluar dari LA Galaxy',
          'Polisi Tangkap Orang yang Mengibarkan Bendera OPM',
          'Abu Vulkanik Merapi Sampai ke daerah Jakarta',
          'No Game No Life Bakal Bikin Season ke 2',
          'Video Porno Aryodj Pimpinan DPR: MKD Punya Kewajiban Meluruskan',
          'Amien Rais Dulu Sempat Mau Bertemu Jokowi tapi Batal',
          'Puan Maharani Sudah Bertemu Prabowo',
          'Megawati Membantah dibully Netizen',
          'Kontrak Pelatih Real Madrid tidak diperpanjang',
          'Chandra Liaw Bertengkar dengan TommyLimmm',
          'LA Galaxy menaikan gaji agar Zlatan tidak Keluar',
          'OPM ditembak Mati Polisi',
          'Gunung Merapi Meletus Lagi Menenyemburkan Abu Vulkanik',
          'Season Ke 2 Pemeran Utama Akan Menjadi Jahat')
corpus <- VCorpus(VectorSource(data))

# Create a document term matrix.
tdm <- DocumentTermMatrix(corpus, list(removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))

# Convert to a data.frame for training and assign a classification (factor) to each document.
train <- as.matrix(tdm)
train <- cbind(train, c(0, 1))
colnames(train)[ncol(train)] <- 'y'
train <- as.data.frame(train)
train$y <- as.factor(train$y)
data
train
# Train.
fit <- train(y ~ ., data = train, method = 'bayesglm')

# Check accuracy on training.
predict(fit, newdata = train)

# Test data.
data2 <- c('Heboh Beredar Video Porno Mirip Anggota DPR Aryodj',
           'Amien Ingin Ketemu Jokowi di Luar Istana Itu Biasa',
           'Puan Ketemu Langsung sama Prabowo',
           'Megawati Sesak Napas Sering dibully Netizen',
           'Zidane tidak Lagi Melatih Real',
           'Tommy dan Chandra membantah Bakal Pisah',
           'Zlatan akan tetap Bermain di LA Galaxy',
           'OPM demo Didepan Polda atas Tembakan yang Terjadi',
           'Abu Vulkanik Menutupi Jogja',
           'Season Ke 2 No Game No Life Tidak akan Keluar')
corpus <- VCorpus(VectorSource(data2))
tdm <- DocumentTermMatrix(corpus, control = list(dictionary = Terms(tdm), removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))
test <- as.matrix(tdm)

# Check accuracy on test.
predict(fit, newdata = test)

