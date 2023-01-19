def create_window(data, window_size, horizon = 1):
        out = []
        L = len(data)

        for i in range(L - window_size - horizon):
            window = data[i : i+window_size, :]
            label = data[i+window_size : i+window_size+horizon, :]
            out.append((window,label))
        return out