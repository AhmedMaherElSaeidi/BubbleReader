from BubbleReader import BubbleReader

reader = BubbleReader()
generated_csv = reader.read_bubblesheet('./public/sample (1).jpeg')
print(generated_csv)