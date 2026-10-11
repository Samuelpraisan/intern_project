{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1KGObjrmMW3Scyy3tH1M4DKR7FL9cNx75",
      "authorship_tag": "ABX9TyNF0+gvpA4/ry5dFg6eQ/1r",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Samuelpraisan/intern_project/blob/main/titanic_ml.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SmrbyyTUI1ZS",
        "outputId": "1a8e484c-c11b-4f80-fb62-74e78953cb37"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2026-10-05 17:59:23.768 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.922 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.13/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n",
            "2026-10-05 17:59:23.923 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.924 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.926 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.928 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.930 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.931 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.933 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.936 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.937 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.938 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.940 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.942 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.943 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.944 Session state does not function when running a script without `streamlit run`\n",
            "2026-10-05 17:59:23.946 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.947 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.948 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.949 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.950 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.951 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.952 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.954 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.955 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.955 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.956 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.957 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.958 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.959 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.961 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.961 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.962 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-10-05 17:59:23.963 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ],
      "source": [
        "import streamlit as st\n",
        "import pickle\n",
        "import pandas as pd\n",
        "\n",
        "with open(\"model.pkl\", \"rb\") as file:\n",
        "    model = pickle.load(file)\n",
        "\n",
        "st.title(\"Titanic Survival Prediction\")\n",
        "\n",
        "st.write(\"Enter passenger details to predict survival.\")\n",
        "\n",
        "pclass = st.slider(\n",
        "    \"Passenger Class\",\n",
        "    1,\n",
        "    3,\n",
        "    1\n",
        ")\n",
        "\n",
        "sex = st.selectbox(\n",
        "    \"Gender\",\n",
        "    [\"Male\", \"Female\"]\n",
        ")\n",
        "\n",
        "age = st.slider(\n",
        "    \"Age\",\n",
        "    1,\n",
        "    80,\n",
        "    25\n",
        ")\n",
        "\n",
        "fare = st.slider(\n",
        "    \"Fare\",\n",
        "    0.0,\n",
        "    500.0,\n",
        "    50.0\n",
        ")\n",
        "\n",
        "if st.button(\"Predict\"):\n",
        "\n",
        "    gender = 0 if sex == \"Male\" else 1\n",
        "\n",
        "    input_data = pd.DataFrame({\n",
        "        \"pclass\": [pclass],\n",
        "        \"sex\": [gender],\n",
        "        \"age\": [age],\n",
        "        \"fare\": [fare]\n",
        "    })\n",
        "\n",
        "    prediction = model.predict(input_data)\n",
        "\n",
        "    if prediction[0] == 1:\n",
        "        st.success(\"Prediction: Passenger is likely to Survive\")\n",
        "    else:\n",
        "        st.error(\"Prediction: Passenger is unlikely to Survive\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!ls -l app.py model.pkl\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "37i83k9XOZiY",
        "outputId": "7d90be9b-5c53-4871-aa7e-3ad3b201de65"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "-rw-r--r-- 1 root root   795 Oct  5 17:43 app.py\n",
            "-rw-r--r-- 1 root root 27740 Oct  5 17:58 model.pkl\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!streamlit run app.py --server.port 8501 > /content/streamlit.log 2>&1 &"
      ],
      "metadata": {
        "id": "79AVgce0M3VF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!cat /content/streamlit.log"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7x7yIHZuO2f6",
        "outputId": "e5c6af2b-5340-4e10-fda6-aa4d7b9536a5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.\n",
            "\n",
            "2026-10-05 18:02:33.406 Uvicorn server started on :::8501\n",
            "\n",
            "  You can now view your Streamlit app in your browser.\n",
            "\n",
            "  Local URL: http://localhost:8501\n",
            "  Network URL: http://172.28.0.12:8501\n",
            "  External URL: http://136.65.15.4:8501\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!cloudflared tunnel --url http://localhost:8501"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6LY5n94wO42s",
        "outputId": "92b2b0fa-e5b9-4543-a867-0dce6de61edb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/bin/bash: line 1: cloudflared: command not found\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "!npm install -g localtunnel"
      ],
      "metadata": {
        "id": "9ZJzNLnoNEas"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import seaborn as sns\n",
        "import pickle\n",
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "\n",
        "df = sns.load_dataset(\"titanic\")\n",
        "\n",
        "df = df[[\"survived\", \"pclass\", \"sex\", \"age\", \"fare\"]]\n",
        "\n",
        "df[\"age\"] = df[\"age\"].fillna(df[\"age\"].median())\n",
        "df[\"sex\"] = df[\"sex\"].map({\"male\": 0, \"female\": 1})\n",
        "\n",
        "X = df[[\"pclass\", \"sex\", \"age\", \"fare\"]]\n",
        "y = df[\"survived\"]\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.2, random_state=42\n",
        ")\n",
        "\n",
        "model = DecisionTreeClassifier(random_state=42)\n",
        "\n",
        "model.fit(X_train, y_train)\n",
        "\n",
        "with open(\"model.pkl\", \"wb\") as file:\n",
        "    pickle.dump(model, file)\n",
        "\n",
        "print(\"Model saved successfully!\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nxWG-yDoN1-x",
        "outputId": "a616ec53-1e02-4026-a256-f4bcf44da362"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model saved successfully!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install streamlit\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Iz1pK1YqKaSM",
        "outputId": "1d54b648-88f4-47ca-ead9-3c6754d4b283"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting streamlit\n",
            "  Downloading streamlit-1.65.0-py3-none-any.whl.metadata (10 kB)\n",
            "Requirement already satisfied: altair!=5.4.0,!=5.4.1,<7,>=5.0.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (8.5.0)\n",
            "Requirement already satisfied: numpy<3,>=1.23 in /usr/local/lib/python3.13/dist-packages (from streamlit) (2.1.3)\n",
            "Requirement already satisfied: packaging>=20 in /usr/local/lib/python3.13/dist-packages (from streamlit) (26.3)\n",
            "Requirement already satisfied: pandas<4,>=1.4.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (2.2.3)\n",
            "Requirement already satisfied: pillow<13,>=7.1.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (11.3.0)\n",
            "Collecting pydeck<1,>=0.8.0b4 (from streamlit)\n",
            "  Downloading pydeck-0.9.3-py2.py3-none-any.whl.metadata (4.2 kB)\n",
            "Requirement already satisfied: protobuf<8,>=5.26.1 in /usr/local/lib/python3.13/dist-packages (from streamlit) (6.33.6)\n",
            "Requirement already satisfied: pyarrow!=25.0.0,<26,>=7.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (23.0.1)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.13/dist-packages (from streamlit) (2.32.4)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.13/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.10.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (4.16.0)\n",
            "Requirement already satisfied: starlette<2,>=0.46.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (1.7.0)\n",
            "Requirement already satisfied: uvicorn<1,>=0.30.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (0.54.0)\n",
            "Requirement already satisfied: httptools<1,>=0.6.3 in /usr/local/lib/python3.13/dist-packages (from streamlit) (0.8.0)\n",
            "Requirement already satisfied: anyio<5,>=4.0.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (4.15.1)\n",
            "Requirement already satisfied: python-multipart<1,>=0.0.10 in /usr/local/lib/python3.13/dist-packages (from streamlit) (0.0.32)\n",
            "Requirement already satisfied: websockets<18,>=12.0.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (15.0.1)\n",
            "Requirement already satisfied: itsdangerous<3,>=2.1.2 in /usr/local/lib/python3.13/dist-packages (from streamlit) (2.2.0)\n",
            "Requirement already satisfied: watchdog<7,>=2.1.5 in /usr/local/lib/python3.13/dist-packages (from streamlit) (6.0.0)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.13/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (3.1.6)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.13/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (4.26.0)\n",
            "Requirement already satisfied: narwhals>=1.14.2 in /usr/local/lib/python3.13/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (2.26.0)\n",
            "Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.13/dist-packages (from anyio<5,>=4.0.0->streamlit) (3.20)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.13/dist-packages (from pandas<4,>=1.4.0->streamlit) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.13/dist-packages (from pandas<4,>=1.4.0->streamlit) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.13/dist-packages (from pandas<4,>=1.4.0->streamlit) (2026.4)\n",
            "Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.13/dist-packages (from requests<3,>=2.27->streamlit) (3.4.9)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.13/dist-packages (from requests<3,>=2.27->streamlit) (2.5.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.13/dist-packages (from requests<3,>=2.27->streamlit) (2026.7.22)\n",
            "Requirement already satisfied: h11>=0.8 in /usr/local/lib/python3.13/dist-packages (from uvicorn<1,>=0.30.0->streamlit) (0.16.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.13/dist-packages (from jinja2->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (3.0.3)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.13/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (26.1.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.13/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (2025.9.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.13/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (0.37.0)\n",
            "Requirement already satisfied: rpds-py>=0.25.0 in /usr/local/lib/python3.13/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (2026.6.3)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.13/dist-packages (from python-dateutil>=2.8.2->pandas<4,>=1.4.0->streamlit) (1.17.0)\n",
            "Downloading streamlit-1.65.0-py3-none-any.whl (10.3 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m10.3/10.3 MB\u001b[0m \u001b[31m74.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading pydeck-0.9.3-py2.py3-none-any.whl (11.4 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m11.4/11.4 MB\u001b[0m \u001b[31m85.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: pydeck, streamlit\n",
            "Successfully installed pydeck-0.9.3 streamlit-1.65.0\n"
          ]
        }
      ]
    }
  ]
}