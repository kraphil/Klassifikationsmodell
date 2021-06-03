from pm4py.objects.conversion.log import converter as xes_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd

log = xes_importer.apply('xyz.xes')
csvpd = xes_converter.apply(log, variant=xes_converter.Variants.TO_DATA_FRAME)

df = pd.DataFrame(csvpd)
df.to_csv(r'xyz.csv', index=False)