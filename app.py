import re
from collections import Counter
import fitz
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="AChemS Research Atlas", page_icon="🧬", layout="wide")
st.markdown("""<style>.block-container{max-width:1400px;padding-top:1.5rem}h1,h2{letter-spacing:-.03em}</style>""",unsafe_allow_html=True)
DRIVE="https://drive.google.com/drive/folders/1EVbIZI7RNa5dd4HncUhwaBS062l7-l6i"
FIELDS=["year","title","authors","abstract","institution","presentation_type","source_file","source_page","verified"]
TOPICS={
"Olfaction & odor perception":r"olfact|odor|odour|smell|aroma|odorant",
"Taste & gustation":r"gustat|taste|tastant|bitter|sweet|umami",
"Receptors & signaling":r"receptor|signal transduct|g.protein|channel",
"Genetics & molecular biology":r"gene|genom|genetic|rna|dna|molecular",
"Neuroscience & brain":r"neuron|neural|brain|cortex|bulb|synap",
"Clinical & human health":r"patient|clinical|disease|anosmia|covid|parosmia",
"Behavior & cognition":r"behavio|learning|memory|cognit|emotion",
"Development & aging":r"develop|aging|ageing|infant|child"}
def clean(df):
    df=df.copy()
    for k in FIELDS:
        if k not in df: df[k]=""
    df["year"]=pd.to_numeric(df["year"],errors="coerce")
    df=df.dropna(subset=["year"])
    df["year"]=df["year"].astype(int)
    df=df[df.year.between(1979,2026)]
    for k in FIELDS[1:]:df[k]=df[k].fillna("").astype(str)
    return df[FIELDS].drop_duplicates(subset=["year","title","source_file","source_page"])
def category(s):
    for name,pat in TOPICS.items():
        if re.search(pat,str(s),re.I):return name
    return "Other / unclassified"
def extract(file):
    m=re.search(r"(?:19|20)\d{2}",file.name)
    if not m:return [],{"file":file.name,"status":"Year missing from filename","candidates":0}
    year=int(m.group());doc=fitz.open(stream=file.getvalue(),filetype="pdf");out=[];readable=0
    for p,page in enumerate(doc,1):
        raw=page.get_text()
        if len(raw.strip())<50:continue
        readable+=1
        for s in raw.splitlines():
            s=re.sub(r"\s+"," ",s).strip()
            if not (28<=len(s)<=190 and 5<=len(s.split())<=26):continue
            if not re.search("[A-Za-z]",s) or re.search(r"https?://|www\.|@|\.edu\b",s,re.I):continue
            if re.search(r"^(abstracts?|session|lunch|registration|break|annual meeting|page \d|\d{1,2}:\d{2})\b",s,re.I):continue
            out.append({"year":year,"title":s,"authors":"","abstract":"","institution":"","presentation_type":"candidate","source_file":file.name,"source_page":p,"verified":"No"})
    doc.close()
    return out,{"file":file.name,"status":"Candidates require manual review" if readable else "Scanned PDF — OCR needed","candidates":len(out)}
if "records" not in st.session_state:st.session_state.records=pd.DataFrame(columns=FIELDS)
if "report" not in st.session_state:st.session_state.report=[]
df=clean(st.session_state.records)
@st.cache_data
def get_2025_olfaction():
    try:
        return pd.read_csv("data/olfaction_2025.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()

st.header("Explore real conference presentations")
selected_year=st.selectbox("Meeting year",list(range(2026,1978,-1)),index=1,key="explore_year")
selected_topic=st.selectbox("Research topic",["Olfaction"],key="explore_topic")
if selected_year==2025:
    source_records=get_2025_olfaction()
    if source_records.empty:
        st.warning("The 2025 source index is not available yet.")
    else:
        st.success(f"{len(source_records)} extracted 2025 olfaction-related title matches from the original AChemS program.")
        search=st.text_input("Search matching presentation titles or authors",key="olf_search")
        if search:
            mask=source_records["title"].str.contains(search,case=False,regex=False,na=False)|source_records["authors"].str.contains(search,case=False,regex=False,na=False)
            source_records=source_records[mask]
        for _,item in source_records.iterrows():
            with st.expander(f"#{item['number']} — {item['title']}"):
                st.write("**Extracted author line:**",item["authors"])
                st.caption("Automatically extracted from the 2025 conference program; title and author formatting require verification.")
                st.link_button("Open original 2025 program and abstracts",item["source"])
else:
    st.info("2025 Olfaction is available now. Other years and topics will be added after source extraction and validation.")
st.divider()

st.title("🧬 AChemS Research Atlas")
st.caption("Historical conference research • 1979–2026 • Exploratory research prototype")
st.info("Charts reflect imported records only, not verified historical findings. PDF extraction yields candidate text that must be checked against the original programs.")
archive,importer,timeline,compare,topics,network,records=st.tabs(["📚 Archive","📥 Import PDFs","📈 Timeline","⚖️ Compare years","🧪 Research topics","🕸️ Collaboration","🗃️ Records"])
with archive:
    a,b,c=st.columns(3)
    a.metric("Years with source PDFs",48);b.metric("Years in imported records",df.year.nunique());c.metric("Imported records",len(df))
    st.write("The connected project archive contains 80 PDF files spanning 1979–2026. This is **file coverage**, not a count of conference presentations.")
    st.dataframe(pd.DataFrame({"Year":range(1979,2027),"Archive status":["PDF available"]*48}),hide_index=True,use_container_width=True,height=280)
    st.link_button("Open source PDFs in Google Drive",DRIVE)
    st.warning("Some early files need OCR. The 2008 and 2020 source files are labeled ISOT.")
with importer:
    st.subheader("Import original conference programs")
    st.write("Upload PDF files downloaded from the AChemS Google Drive, or a reviewed CSV. This website does not automatically read private Drive files.")
    pdfs=st.file_uploader("Upload one or more PDFs",type="pdf",accept_multiple_files=True)
    if st.button("Extract candidate text",disabled=not bool(pdfs),type="primary"):
        rows=[];reports=[]
        for f in pdfs:
            try:
                part,report=extract(f);rows+=part;reports.append(report)
            except Exception as e:reports.append({"file":f.name,"status":"Could not parse PDF: "+type(e).__name__,"candidates":0})
        if rows:st.session_state.records=clean(pd.concat([st.session_state.records,pd.DataFrame(rows)],ignore_index=True))
        st.session_state.report=reports
        st.success(f"Extracted {len(rows)} unverified candidate lines.")
    if st.session_state.report:st.dataframe(pd.DataFrame(st.session_state.report),hide_index=True,use_container_width=True)
    csv=st.file_uploader("Or upload reviewed CSV",type="csv")
    if st.button("Load CSV",disabled=csv is None):
        try:
            incoming=pd.read_csv(csv)
            if not {"year","title"}.issubset(incoming.columns):st.error("CSV needs year and title columns.")
            else:
                st.session_state.records=clean(pd.concat([st.session_state.records,incoming],ignore_index=True));st.success("CSV loaded.")
        except Exception as e:st.error(str(e))
with timeline:
    st.subheader("Imported records by meeting year")
    if df.empty:st.info("Import data to view trends.")
    else:
        d=df.groupby("year").size().reset_index(name="records")
        st.plotly_chart(px.line(d,x="year",y="records",markers=True),use_container_width=True)
        st.caption("Do not interpret unverified candidate counts as actual conference presentation counts.")
with compare:
    st.subheader("Compare two years")
    a,b=st.columns(2);y1=a.selectbox("First year",range(1979,2027));y2=b.selectbox("Second year",range(1979,2027),index=47)
    l,r=st.columns(2)
    for col,year in [(l,y1),(r,y2)]:
        with col:
            subset=df[df.year==year];st.markdown(f"### {year}");st.metric("Imported records",len(subset))
            if subset.empty:st.info("No records imported for this year.")
            else:
                t=subset.title.map(category).value_counts().reset_index();t.columns=["Topic","Count"]
                st.plotly_chart(px.bar(t,x="Count",y="Topic",orientation="h"),use_container_width=True)
                st.dataframe(subset[["title","verified"]].head(30),hide_index=True,use_container_width=True)
with topics:
    st.subheader("Exploratory research themes")
    if df.empty:st.info("Import records to begin.")
    else:
        d=df.copy();d["topic"]=(d.title+" "+d.abstract).map(category)
        c=d.groupby(["year","topic"]).size().reset_index(name="n");c["share"]=c.n/c.groupby("year").n.transform("sum")*100
        st.plotly_chart(px.line(c,x="year",y="share",color="topic",markers=True,title="Preliminary keyword category shares (%)"),use_container_width=True)
        with st.expander("Experiment with unsupervised title clustering"):
            k=st.slider("Number of clusters",2,12,5)
            if st.button("Cluster titles"):
                if len(d)<k:st.warning("Not enough records.")
                else:
                    try:
                        x=TfidfVectorizer(stop_words="english",ngram_range=(1,2)).fit_transform(d.title)
                        d["cluster"]=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(x)
                        st.dataframe(d[["year","title","cluster"]].sort_values("cluster"),hide_index=True,use_container_width=True)
                    except ValueError as e:st.warning(str(e))
        st.caption("Topics and clusters require expert validation before research publication.")
with network:
    st.subheader("Coauthor collaboration network")
    authored=df[df.authors.str.strip().ne("")]
    if authored.empty:st.info("Upload a CSV containing an authors column with names separated by semicolons.")
    else:
        miny,maxy=int(authored.year.min()),int(authored.year.max())
        span=st.slider("Years",miny,maxy,(miny,maxy)) if miny<maxy else (miny,maxy)
        G=nx.Graph()
        for s in authored[authored.year.between(*span)].authors:
            people=list(dict.fromkeys(p.strip() for p in s.split(";") if p.strip()))
            G.add_nodes_from(people)
            for i,a in enumerate(people):
                for b in people[i+1:]:G.add_edge(a,b)
        st.write(f"**{G.number_of_nodes()} authors** · **{G.number_of_edges()} coauthor links**")
        if G.number_of_nodes():
            keep=sorted(G.degree,key=lambda p:p[1],reverse=True)[:60];H=G.subgraph([x for x,_ in keep])
            pos=nx.spring_layout(H,seed=17);ex=[];ey=[]
            for a,b in H.edges:
                ex.extend([pos[a][0],pos[b][0],None]);ey.extend([pos[a][1],pos[b][1],None])
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=ex,y=ey,mode="lines",line=dict(color="#aab6c3",width=1),hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[pos[p][0] for p in H],y=[pos[p][1] for p in H],mode="markers+text",text=list(H.nodes),textposition="top center",marker=dict(size=11,color="#2477a7")))
            fig.update_layout(showlegend=False,height=650,xaxis=dict(visible=False),yaxis=dict(visible=False))
            st.plotly_chart(fig,use_container_width=True)
            st.caption("Coauthor names are not disambiguated; network is preliminary.")
with records:
    st.subheader("Review, correct and export")
    if df.empty:st.info("No imported records.")
    else:
        edited=st.data_editor(df,num_rows="dynamic",hide_index=True,use_container_width=True)
        if st.button("Save changes in this session"):st.session_state.records=clean(edited);st.success("Saved for this session.")
        st.download_button("Download reviewed CSV",clean(edited).to_csv(index=False).encode(),file_name="achems_records.csv",mime="text/csv")
        st.caption("Export before closing; session data may reset. Set verified to Yes only after source review.")
st.divider()
st.caption("Independent AChemS historical research prototype · Source records require validation.")
