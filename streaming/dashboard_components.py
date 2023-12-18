


def display_streaming_dashboard():
    st.title("Streaming Trading Dashboard")
    # Continuously update the live plot and transactions table
    while True:
        # Make sure to lock the data when accessing it
        with st.session_state.data_lock:
            if not st.session_state.streamed_data.empty:
                live_data_placeholder.line_chart(st.session_state.streamed_data.set_index('time')['price'])
            if not st.session_state.transactions.empty:
                transactions_placeholder.table(st.session_state.transactions.head(10))
        # Sleep briefly to allow other threads to run and to prevent constant updates
        time.sleep(0.1)

    # Sidebar
    with st.sidebar:
        st.header('Controls')
        technical_indicators = st.multiselect('Add to plot:', ['SMA', 'EMA', 'RSI'])

        # Input for target profit
        target_profit = st.text_input('Target Profit')

        # Dropdown for selecting coins
        selected_coin = st.selectbox('Select Coin', ['BTC', 'ETH', 'SHIB'])

        # Toggle for autotrade
        if st.button('Toggle Autotrade'):
            if st.session_state.trading_on:
                stop_trading()
            else:
                start_trading()
            st.session_state.trading_on = not st.session_state.trading_on

    # Analysis Dashboard
    with st.container():
        st.header('Analysis Dashboard')
        # Input for target profit
        target_profit = st.text_input('Target Profit')

        # # Dropdown for selecting coins
        selected_coin = st.selectbox('Select Coin', ['BTC', 'ETH', 'SHIB'])


    with st.container():
        # Dynamic plot for streamed data
        live_data_placeholder = st.empty()
        # Dynamic table for transactions
        transactions_placeholder = st.empty()



    with st.container():
        # Time series plot for past data
        # st.line_chart(np.random.randn(100, 2))  # Placeholder for actual data

        # Time interval discretization
        discretization_interval = st.slider('Discretization Interval:', 5, 60, 5)

        # Profit updates
        if st.session_state.transactions.empty:
            st.write("No transactions made yet.")
        else:
            st.write(f"Latest Profit: {st.session_state.transactions.iloc[0]['Profit']}")

    # Table of transaction details
    with st.container():
        st.header('Transaction Details')
        if not st.session_state.transactions.empty:
            st.table(st.session_state.transactions.head(10))

    # Generate report button and area
    with st.container():
        if st.button('Generate Report'):
            st.session_state.report_generated = generate_report()
        st.text_area('Report', st.session_state.report_generated if 'report_generated' in st.session_state else '')

