
import streamlit as st
import pandas as pd
from prophet import Prophet

st.set_page_config(page_title="پیش‌بینی قیمت طلا", layout="centered")
st.title("اپلیکیشن تحت وب پیش‌بینی قیمت طلا با هوش مصنوعی")

st.write("""
لطفاً فایل CSV قیمت طلا را بارگذاری کنید. 
فایل باید دو ستون داشته باشد: `تاریخ` و `قیمت` (به میلادی یا شمسی که قابل تبدیل باشد)
""")

uploaded_file = st.file_uploader("بارگذاری فایل CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = ["ds", "y"]

    st.success("فایل با موفقیت خوانده شد!")
    st.write("پیش‌نمایش داده‌ها:")
    st.dataframe(df.tail())

    with st.spinner("در حال آموزش مدل هوش مصنوعی..."):
        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=5)
        forecast = model.predict(future)

    st.success("مدل آموزش دید! حالا پیش‌بینی ۵ روز آینده را ببین:")
    st.dataframe(forecast[['ds', 'yhat']].tail())

    st.line_chart(forecast.set_index('ds')[['yhat']], use_container_width=True)
else:
    st.warning("منتظر بارگذاری فایل CSV هستیم...")
