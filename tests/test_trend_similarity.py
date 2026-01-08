from cv_engine.trend_similarity import TrendSimilarity

ts = TrendSimilarity()

# step 1—add a sample to viral bank
ts.add_to_bank("bus.jpg", label="example")

# step 2—compute similarity
score = ts.similarity_score("bus.jpg")
print("Trend similarity score:", score)
