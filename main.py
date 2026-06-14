import streamlit as st
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

from yaml.loader import SafeLoader
from joblib import load
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher

if "pred" not in st.session_state:
    st.session_state["pred"] = None

with open("config.yaml") as file:
    config = yaml.load(stream=file, Loader=SafeLoader)

Hasher.hash_passwords(credentials=config["credentials"])

authenticator = stauth.Authenticate(
    credentials=config["credentials"],
    cookie_name=config["cookie"]["name"],
    cookie_key=config["cookie"]["key"],
    cookie_expiry_days=config["cookie"]["expiry_days"]
)

authenticator.login(location="main")
st.divider()
with st.expander("Register"):
    try:
        if authenticator.register_user(location="main", clear_on_submit=True):
            st.success('User registered successfully!')

            for username, data in config['credentials']['usernames'].items():
                if 'roles' in data and data['roles'] is None:
                    del data['roles']

            with open('config.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
    except Exception as e:
        st.error(e)


@st.cache_data
def read_data():
    df = pd.read_csv('bds_part_abc_cleansed.csv')
    return df

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model = load('bds_random_forest.joblib')
    return model

def make_prediction(model):
    X_pred = pd.DataFrame({
        "area": [st.session_state["area"]],
        "number_of_bedrooms": [st.session_state["bedrooms"]],
        "number_of_bathrooms": [st.session_state["bathrooms"]],
        "number_of_stories": [st.session_state["stories"]],
        "front_length": [st.session_state["front"]],
        "legal_status": [True if st.session_state["legal"] == "Sổ đỏ" else False],
        "old_address": [st.session_state["district"]]
    })

    pred = model.predict(X_pred)
    pred = round(pred[0], 2)
    st.session_state["pred"] = pred

def display_chart(df: pd.DataFrame):
    fig, ax = plt.subplots()
    match st.session_state["column_visualization"]:
        case "Giá":
            sns.histplot(data=df, x="price_range", kde=True, color="blue")
            plt.title("Distribution of Price")
            plt.xlabel("Billions VND")
            plt.ylabel("Frequency")
        case "Diện tích":
            sns.histplot(data=df, x="area", kde=True, color="blue")
            plt.title("Distribution of Area")
            plt.xlabel("Area (m2)")
            plt.ylabel("Frequency")
        case "Giá/m2":
            sns.histplot(data=df, x="price_per_m2", kde=True, color="blue")
            plt.title("Distribution of Price per m2")
            plt.xlabel("Price per m2 (Millions VND/m2)")
            plt.ylabel("Frequency")
        case "Tầng":
            sns.countplot(data=df, y="number_of_stories", order=df["number_of_stories"].value_counts(ascending=True).index, color="blue")
            plt.title("Distribution of Number of stories")
            plt.xlabel("Frequency")
            plt.ylabel("Number of stories")
        case "Phòng ngủ":
            sns.countplot(data=df, y="number_of_bedrooms", order=df["number_of_bedrooms"].value_counts(ascending=True).index, color="blue")
            plt.title("Count of Number of bedrooms")
            plt.xlabel("Frequency")
            plt.ylabel("Number of bedrooms")
        case _:
            sns.countplot(data=df, y="number_of_bathrooms", order=df["number_of_bathrooms"].value_counts(ascending=True).index, color="blue")
            plt.title("Count of Number of bathrooms")
            plt.xlabel("Frequency")
            plt.ylabel("Number of bathrooms")

    st.pyplot(fig)

if st.session_state["authentication_status"]:
    authenticator.logout(location="sidebar")
    st.title("House price calculator")

    df = read_data()
    reg = load_model()

    with st.form(key="form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.number_input(label="Diện tích", min_value=1.0, value=50.0, step=0.1, key="area")
            st.number_input(label="Tầng", min_value=1, value=5, step=1, key="stories")
            st.selectbox(label="Quận/Huyện", index=0, key="district",
                         options=['Tây Hồ', 'Hà Đông', 'Ba Đình', 'Hoàn Kiếm', 'Thanh Trì', 'Cầu Giấy',
                                 'Hoài Đức', 'Thanh Xuân', 'Nam Từ Liêm', 'Hoàng Mai', 'Long Biên', 'Đống Đa',
                                 'Đông Anh', 'Thanh Oai', 'Hai Bà Trưng', 'Gia Lâm', 'Bắc Từ Liêm', 'Chương Mỹ',
                                 'Quốc Oai', 'Thường Tín', 'Phúc Thọ', 'Đan Phượng', 'Ứng Hòa', 'Mê Linh',
                                 'Mỹ Đức', 'Sóc Sơn', 'Thạch Thất'])

        with col2:
            st.number_input(label="Mặt tiền", min_value=1.0, value=4.5, step=0.1, key="front")
            st.number_input("Phòng ngủ", min_value=1, value=4, step=1, key="bedrooms")
        with col3:
            st.selectbox("Pháp lý", options=['Sổ đỏ', 'Không có sổ'], index=0, key="legal")
            st.number_input("Phòng tắm", min_value=1, value=4, key="bathrooms")

        st.form_submit_button("Tính toán", type="primary", on_click=make_prediction, kwargs=dict(model=reg))

    if st.session_state["pred"] is not None:
        st.subheader(f"Giá ước lượng cho căn nhà trên là {st.session_state.pred} tỷ VND")
    else:
        st.write("Nhập thông tin và nhấn nút Tính toán để có ước lượng giá căn nhà")

    st.write("\n------------------------------------------------------\n")
    with st.expander(label="Xem dữ liệu"):
        st.dataframe(df)

    st.write("\n------------------------------------------------------\n")
    st.subheader("Columns Visualization")
    st.selectbox("Chọn cột", options=["Giá", "Diện tích", "Giá/m2", "Tầng", "Phòng ngủ", "Phòng tắm"], index=0, key="column_visualization")
    display_chart(df)

elif st.session_state["authentication_status"] == False:
    st.error("Nhập sai tài khoản hoặc mật khẩu")
elif st.session_state["authentication_status"] is None:
    pass
