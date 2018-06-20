library(caret)
library(tm)
library(SnowballC)
library(arm)
# Training data.
data <- c('Amien Rais bertemu Jokowi',
          'Jokowi Atur Waktu Temui Amien Rais',
          'Ada kejanggalan Amien saat bertemu Jokowi',
          'Netizen tidak percaya adanya pertemuan Amien dengan Jokowi',
          'Jokowi belum sempat kasih kabar petemuan untuk Amien',
          'Amien Rais batal mengunjungi Jokowi',
          'Belum ada kepastian Amien bertemu Jokowi',
          'Pertemuan yang sudah direncanakan Amien',
          'Jokowi dan Amien jalan bareng',
          'Melihat Jokowi bersama Amien Rais Netizen membuat meme',
          'Ada rencana dibalik pertemuan ini kata Amien',
          'Amien Rais Dulu Sempat Mau Bertemu Jokowi tapi Batal',
          'Hendak bertemu Jokowi',
          'Amien sudah menunggu pertemuan ini',
          'Apa yang dibahas dalam pertemuan Jokowi Amien',
          'Pertemuan ini hanya mencari muka komen netizen',
          'Tidak adanya pertemuan kata Jokowi',
          'Amien rais belum siap bertemu',
          'Percaya atau tidak semua terserah netizen kata Amien',
          'Amien tidak ada kabar mau bertemu Jokowi')
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
data2 <- c('Tidak ada pertemuan Amien dengan Jokowi',
           'Amien Ingin Ketemu Jokowi di Luar Istana Itu Biasa',
           'Pertemuan ini berjalan lancar',
           'Jokowi sempat Selfie bersama Amien Rais saat bertemu',
           'Ada rencana tersembunyi dalam pertemuan Amien kata Netizen',
           'Jokowi mengundurkan waktu pertemuan Amien Rais',
           'Amien kecewa belum sempat bertemu dengan Jokowi',
           'Jokowi hendak bertemu hanya berpapasan',
           'Rencana bertemu Jokowi Amien batal datang',
           'Tidak ada kepastian kapan bertemu')
corpus <- VCorpus(VectorSource(data2))
tdm <- DocumentTermMatrix(corpus, control = list(dictionary = Terms(tdm), removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))
test <- as.matrix(tdm)

# Check accuracy on test.
predict(fit, newdata = test)
