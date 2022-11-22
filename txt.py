fig, axs = plt.subplots(2, 3, figsize=(13,13))
axs[0, 0].scatter(listaVariada["danceability"], listaVariada["energy"], c=listaVariada["class"], alpha=0.5)
axs[0, 0].set_xlabel("danceability")
axs[0, 0].set_ylabel("energy")

axs[0, 1].scatter(listaVariada["danceability"], listaVariada["loudness"], c=listaVariada["class"], alpha=0.5)
axs[0, 1].set_xlabel("danceability")
axs[0, 1].set_ylabel("loudness")

axs[0, 2].scatter(listaVariada["danceability"], listaVariada["tempo"], c=listaVariada["class"], alpha=0.5)
axs[0, 2].set_xlabel("danceability")
axs[0, 2].set_ylabel("tempo")

axs[1, 0].scatter(listaVariada["energy"], listaVariada["loudness"], c=listaVariada["class"], alpha=0.5)
axs[1, 0].set_xlabel("sep_wi")
axs[1, 0].set_ylabel("pe_len")

axs[1, 1].scatter(listaVariada["energy"], listaVariada["tempo"], c=listaVariada["class"], alpha=0.5)
axs[1, 1].set_xlabel("energy")
axs[1, 1].set_ylabel("tempo")

axs[1, 2].scatter(listaVariada["loudness"], listaVariada["tempo"], c=listaVariada["class"], alpha=0.5)
axs[1, 2].set_xlabel("loudness")
axs[1, 2].set_ylabel("tempo")