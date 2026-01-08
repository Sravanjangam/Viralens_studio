from cv_engine.aesthetic import AestheticScorer

score = AestheticScorer().score("https://ultralytics.com/images/zidane.jpg")
print("Aesthetic score:", score)
