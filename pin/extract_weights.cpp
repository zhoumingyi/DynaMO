#include <iostream>
#include <fstream>
#include "pin.H"
using std::cerr;
using std::endl;
using std::ios;
using std::ofstream;
using std::string;

/* Names of functions */
// constexpr auto FUNCNUM = 1;
// char *funcNameVec[FUNCNUM] = {"xiiphz::GetDepthConvTensor"};

/* Global Variables */
std::ofstream TraceFile;
// std::vector<std::string> related_func;
std::string OpName;

typedef struct TfLiteIntArray {
  int size;

  int data[];

} TfLiteIntArray;

int str_incl(const CHAR* str1, const CHAR* str2)
{
    if(strstr(str2, str1) == NULL)
        return 1;
    else
        return 0;
}
/* Commandline Switches */
KNOB<std::string> KnobOpName(KNOB_MODE_WRITEONCE, "pintool", "opname", "default", "Specify the operation name");

/* Analysis routines */
VOID beforeCall(CHAR* OpName, CHAR* filter_name, TfLiteIntArray* dim, CHAR* tensor)
{
    float* float_tensot = reinterpret_cast<float*>(tensor);
    TraceFile << OpName << "(";
    TraceFile << filter_name << ", ";

    int num_elements = 1;
    for (int i = 0; i < dim->size; i++)
    {
        num_elements *= dim->data[i];
        TraceFile << dim->data[i] << ", ";
    }

    TraceFile << ")" << endl;

    TraceFile << "(";
    for (int i = 0; i < num_elements; i++)
    {
        TraceFile << float_tensot[i] << ",";
    }
    TraceFile << ")" << endl;
}

/* Print return value of the function. */
VOID afterCall(ADDRINT ret)
{
    // TraceFile << "  return: " << ret << endl;
}

/* Instrument the functions */
VOID InstFunc(IMG& img, SYM& sym, const char* OpName, AFUNPTR bfFunPtr, AFUNPTR afFunPtr)
{
    RTN funcRtn = RTN_FindByAddress(IMG_LowAddress(img) + SYM_Value(sym));
    if (RTN_Valid(funcRtn))
    {
        RTN_Open(funcRtn);
        // get function's argument
        RTN_InsertCall(
            funcRtn,
            IPOINT_BEFORE, bfFunPtr,
            IARG_ADDRINT, OpName,
            // IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
            IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
            IARG_FUNCARG_ENTRYPOINT_VALUE, 2,
            // IARG_FUNCARG_ENTRYPOINT_VALUE, 3,
            IARG_FUNCARG_ENTRYPOINT_VALUE, 4,
            IARG_END);
        RTN_Close(funcRtn);
    }
}

/* Instrumentation routines */
VOID Image(IMG img, VOID* V )
{
    // Walk through the symbols in the symbol table.
    for (SYM sym = IMG_RegsymHead(img); SYM_Valid(sym); sym = SYM_Next(sym))
    {
        string undFuncName = PIN_UndecorateSymbolName(SYM_Name(sym), UNDECORATION_NAME_ONLY);

        std::string FuncName = OpName;
        FuncName.append("::Get");
        if (!str_incl(FuncName.c_str(), undFuncName.c_str()) && !str_incl("Tensor", undFuncName.c_str()))
        {
            // const char* funcName;
            // funcName = strdup(undFuncName.c_str());
            InstFunc(img, sym, OpName.c_str(), (AFUNPTR)beforeCall, (AFUNPTR)afterCall);
        }
        // }
    }
}

VOID Fini(INT32 code, VOID* v)
{
    TraceFile.close();
}

/* Print Help Message */
INT32 Usage()
{
    cerr << "This tool produces a trace of calls to functions." << endl;
    cerr << endl << KNOB_BASE::StringKnobSummary() << endl;
    return -1;
}

/* Main */
int main(int argc, char* argv[])
{
    // Initialize pin & symbol manager
    PIN_InitSymbols();
    if (PIN_Init(argc, argv))
    {
        return Usage();
    }

    OpName = KnobOpName.Value();
    std::cout << "OpName: " << OpName << std::endl;

    // Write to a file since cout and cerr maybe closed by the application
    TraceFile.open("extracted_weights.txt");
    TraceFile << std::dec << std::showbase;

    // Register Image to be called to instrument functions.
    IMG_AddInstrumentFunction(Image, 0);
    PIN_AddFiniFunction(Fini, 0);

    // Never returns
    PIN_StartProgram();

    return 0;
}