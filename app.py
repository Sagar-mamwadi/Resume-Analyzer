import streamlit as st
import pickle as pk
import re
# import nltk

# nltk.download('punkt')
# nltk.download('stopwords')

clf=pk.load(open("clf.pkl","rb"))
tfidf=pk.load(open("tfidf.pkl","rb"))

def cleanResume(text):
    cleantext=re.sub("http\S+\s"," ",text)
    cleantext=re.sub("#\S"," ",cleantext)
    cleantext=re.sub("@\S"," ",cleantext)
    cleantext=re.sub("RT|CC"," ",cleantext)
    cleantext=re.sub('[%s]'%re.escape("""|"#$%&'()"+,-.|:;<=>?@[\]^_`{|}~""")," ",cleantext)
    cleantext=re.sub(r'[^\x00-\x7f]'," ",cleantext)
    cleantext=re.sub("\s+"," ",cleantext)
    return cleantext

def main():
    st.title("Resume Screening App")
    upload_file=st.file_uploader("Upload Resume",type=['txt','pdf','docx'])

    if upload_file is not None:
        try:
            resume_bytes=upload_file.read()
            resume_text=resume_bytes.decode('UTF-8')
        except UnicodeDecodeError:
            resume_text=resume_bytes.decode('latin-1')
        
        cleaned_resume=cleanResume(resume_text)
        cleaned_resume_number=tfidf.transform([cleaned_resume])
        prediction_id=clf.predict(cleaned_resume_number)
        

        category_mapping={
            15:"Java Developer",
            23:"Testing",
            8:"Devops Engineer",
            20:"Python Developer",
            24:"Web Designing",
            12:"HR",
            13:"Hadoop",
            3:"Blockchain",
            10:"ETL Developer",
            18:"Opration Manager",
            6:"Data Science",
            22:"Sales",
            16:"Mechanical Engineer",
            1:"Arts",
            7:"Database",
            11:"Electrical Engineering",
            14:"Health and Fitness",
            19:"PMO",
            4:"Business Analyst",
            9:"Dotnet Developer",
            2:"Automation Testing",
            17:"Network Security Engineer",
            21:"Sap Developer",
            5:"Civil Engineer",
            0:"Advocate",
        }
        category_name=category_mapping.get(int(prediction_id),"unknown")
        st.write("Predicted Category:",category_name)




if __name__ == "__main__":
    main()
