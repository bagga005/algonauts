import model_sklearn
import utils
import train
trainer = model_sklearn.LinearHandler_Sklearn(1650, 1000)
model_name = train.get_model_name(3,'all')
trainer.load_model(model_name)
trainer.plot_coefficient_heatmap(100,200,1599,1650)