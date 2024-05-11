### chendan 2024-5-11 ###
import pandas as pd
import xmnlp

# 设置xmnlp模型路径
print(xmnlp.__version__)
xmnlp.set_model('xmnlp-onnx-models')
# 设置文件路径
file_path = 'corpus_src/pengbei_224_0511.xlsx'
output_file_name = 'label_res_224_0511.xlsx'

def predict_xmnlp(dataframe, sheet):
    result = pd.DataFrame()
    for sen in dataframe.iloc[:, 0]:
        # result = result.append(pd.DataFrame({predict(sen)}))
        result = pd.concat([result, pd.DataFrame({xmnlp.sentiment(sen)})])
    result = result.rename(columns={0:sheet+'neg', 1:sheet+'pos'})
    # print(result.head())
    return result

if __name__ == '__main__':
    # 读取数据
    # df = pd.read_excel(file_path)
    # print(df.head())
    xls = pd.ExcelFile(file_path)
    xls_out = pd.DataFrame()

    # 一次预测
    # fun_res = predict_xmnlp(df, "none")
    # print(fun_res.head())

    # 写入excel表，使用writer才不会覆盖
    with pd.ExcelWriter(output_file_name) as excel_writer:
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            pre = predict_xmnlp(df, sheet_name)
            print(pre.head())
            # xls_out = pd.concat([xls_out, pre], axis=1)
            # xls_out = pd.merge(xls_out, pre, how='cross', left_on=None, right_on=None, left_index=False, right_index=False)
            # xls_out = xls_out.assign({pre})
            # name = sheet_name + '.xlsx'
            pre.to_excel(excel_writer, sheet_name=sheet_name, index=False)


