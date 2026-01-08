from text_engine.caption_analysis import CaptionAnalyzer

analyzer = CaptionAnalyzer()

caption = (
    "Stop scrolling! 🔥 This is the secret to boosting your Reels. "
    "Save this and share with a friend. Link in bio for the full guide. 😎"
)

result = analyzer.analyze(caption)
print("Caption analysis result:")
print(result.to_dict())
